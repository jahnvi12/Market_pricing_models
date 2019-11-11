import sys
sys.path.append('/home/jahnvi/python-rl')

from random import Random
import numpy

import pyrl.basis.fourier as fourier
import pyrl.basis.rbf as rbf
import pyrl.basis.trivial as trivial
from planner import Planner

from sklearn import linear_model
from sklearn.svm import SVR
from sklearn import tree

class FittedQIteration(Planner):
    """FittedQIteration is an implementation of the Fitted Q-Iteration algorithm of Ernst, Geurts, Wehenkel (2005).

    This class allows the use of a variety of regression algorithms, provided by scikits-learn, to be used for
    representing the Q-value function. Additionally, different basis functions can be applied to the features before
    being passed to the regressors, including trivial, fourier, tile coding, and radial basis functions.
    """

    def __init__(self, model, **kwargs):
        """Inits the Fitted Q-Iteration planner with discount factor, instantiated model learner, and additional parameters.

        Args:
            model: The model learner object
            gamma=1.0: The discount factor for the domain
            **kwargs: Additional parameters for use in the class.
        """
        Planner.__init__(self, model, **kwargs)
        self.fa_name = self.params.setdefault('basis', 'trivial')
        self.params.setdefault('iterations', 100)
        self.params.setdefault('support_size', 500)
        self.basis = None
        self.gamma = 0.9
        self.EPSILON = 0.9                
        self.EPSILON_DECAY = 0.9

        # Set up regressor
        learn_name = self.params.setdefault('regressor', 'ridge')
        if learn_name == 'linreg':
            self.learner = linear_model.LinearRegression()
        elif learn_name == 'ridge':
            self.learner = linear_model.Ridge(alpha = self.params.setdefault('l2', 10))
        elif learn_name == 'tree':
            self.learner = tree.DecisionTreeRegressor()
        elif learn_name == 'svm':
            self.learner = SVR()
        else:
            self.learner = None

    def planner_init(self):
        self.has_plan = False
        self.ranges, self.actions = self.model.getStateSpace()
        # Set up basis
        if self.fa_name == 'fourier':
            self.basis = fourier.FourierBasis(len(self.ranges), self.ranges,
                            order=self.params.setdefault('fourier_order', 4))
        elif self.fa_name == 'rbf':
            self.basis = rbf.RBFBasis(len(self.ranges), self.ranges,
                            num_functions=self.params.setdefault('rbf_number', len(self.ranges)),
                            beta=self.params.setdefault('rbf_beta', 1.0))
        else:
            self.basis = trivial.TrivialBasis(len(self.ranges), self.ranges)


    def getStateAction(self, state, action):
        """Returns the basified state feature array for the given state action pair.

        Args:
            state: The array of state features
            action: The action taken from the given state

        Returns:
            The array containing the result of applying the basis functions to the state-action.
        """

        state = self.basis.computeFeatures(state)
        stateaction = numpy.zeros((self.actions, len(state)))
        stateaction[action,:] = state
        return stateaction.flatten()

    def predict(self, state, action):
        """Predict the next state, reward, and termination probability for the current state-action.

        Args:
            state: The array of state features
            action: The action taken from the given state

        Returns:
            Tuple (next_state, reward, termination), where next_state gives the predicted next state,
            reward gives the predicted reward for transitioning to that state, and termination
            gives the expected probabillity of terminating the episode upon transitioning.

            All three are None if no model has been learned for the given action.
        """
        if self.model.has_fit[action]:
            return self.model.predict(state, action)
        else:
            return None, None, None

    def getValue(self, state):
        """Get the Q-value function value for the greedy action choice at the given state (ie V(state)).

        Args:
            state: The array of state features

        Returns:
            The double value for the value function at the given state
        """
        if self.has_plan:
            return self.learner.predict([self.getStateAction(state, a) for a in range(self.actions)]).max()
        else:
            return None

    def getAction(self, state):
        """Get the action under the current plan policy for the given state.

        Args:
            state: The array of state features

        Returns:
            The current greedy action under the planned policy for the given state. If no plan has been formed,
            return a random action.
        """
        if self.has_plan:
            return self.learner.predict([self.getStateAction(state, a) for a in range(self.actions)]).argmax()
        else:
            return self.randGenerator.randint(0, self.actions-1)


    def updatePlan(self):
        """Run Fitted Q-Iteration on samples from the model, and update the plan accordingly."""
        state = self.model.reset()
        for sample_iter in range(3000):
            self.has_plan = False
            prev_coef = None
            samples, outcomes = self.model.sampleStateActions(self.params['support_size'])

            Xp = []
            X = []
            R = []
            gammas = []
            for a in range(self.actions):
                Xp += map(lambda k: [self.getStateAction(k, b) for b in range(self.actions)], outcomes[a][0])
                X += map(lambda k: self.getStateAction(k, a), samples[a])
                R += list(outcomes[a][1])
                gammas += map(lambda k: self.gamma*k, outcomes[a][2])

            Xp = numpy.array(Xp)
            Xp = Xp.reshape(Xp.shape[0]*Xp.shape[1], Xp.shape[2])
            X = numpy.array(X)
            R = numpy.array(R)
            gammas = numpy.array(gammas)
            targets = []
            Qp = None

            error = 1.0
            iter2 = 0
            threshold = 1.0e-10
            while error > threshold and iter2 < self.params['iterations']:
                if self.has_plan:
                    Qprimes = self.learner.predict(Xp).reshape((X.shape[0], self.actions))
                    targets = R + gammas*Qprimes.max(1)
                    Qp = Qprimes
                else:
                    targets = R
                    self.has_plan = True
                self.learner.fit(X, targets)
                try:
                    if prev_coef is not None:
                        error = numpy.linalg.norm(prev_coef - self.learner.coef_)
                    prev_coef = self.learner.coef_.copy()
                except:
                    pass

                iter2 += 1

            Xp = numpy.array([self.getStateAction(state, b) for b in range(self.actions)])
            Qprimes = self.learner.predict(Xp)
            best_action = Qprimes.argmax()

            if self.EPSILON > 0.1:
                    self.EPSILON = self.EPSILON_DECAY * self.EPSILON

            if numpy.random.random() < self.EPSILON:
                best_action = self.model.sample_action()

            state, done = self.model.updateExperience(state, best_action)
            if done:
                state = self.model.reset()

            print (sample_iter, iter2, error, self.model.exp_index)

            if (sample_iter % 100 == 0):
                self.test()
            # if error <= threshold:
            #     return

    def test(self):
        print('Testing starts ====== ')
        state = self.model.reset()
        state[2] = 1
        state[3] = 1
        rewards = [0]*6
        r = [[],[],[],[],[],[]]
        while True:
            Xp = numpy.array([self.getStateAction(state, b) for b in range(self.actions)])
            Qprimes = self.learner.predict(Xp)
            best_action = Qprimes.argmax()
            state, rews, done = self.model.step_2(state, best_action)
            for i in range(6):
                rewards[i]+=rews[i]
                r[i] += [rews[i]]
            if done:
                print(rewards)
                print(r)
                break
