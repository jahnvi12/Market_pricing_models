using PowerModels
using Ipopt
using JuMP


nlp_optimizer = with_optimizer(Ipopt.Optimizer, print_level=0)
# note: print_level changes the amount of solver information printed to the terminal


# Load System Data
# ----------------
powermodels_path = joinpath(dirname(pathof(PowerModels)), "..")

file_name = "$(powermodels_path)/test/data/matpower/case30.m"
# note: change this string to modify the network data that will be loaded

# load the data file
data = PowerModels.parse_file(file_name)

# Add zeros to turn linear objective functions into quadratic ones
# so that additional parameter checks are not required
#PowerModels.standardize_cost_terms!(data, order=2)

# use build_ref to filter out inactive components
ref = PowerModels.build_ref(data)[:nw][0]
# note: ref contains all the relevant system parameters needed to build the OPF model
# When we introduce constraints and variable bounds below, we use the parameters in ref.

function predict(b1_1,b2_1,t)

###############################################################################
# 1. Building the Optimal Power Flow Model
###############################################################################

# Initialize a JuMP Optimization Model
#-------------------------------------
model = Model(nlp_optimizer)


# Add Optimization and State Variables
# ------------------------------------

# Add voltage angles va for each bus
@variable(model, va[i in keys(ref[:bus])])
# Add voltage angles vm for each bus
@variable(model, ref[:bus][i]["vmin"] <= vm[i in keys(ref[:bus])] <= ref[:bus][i]["vmax"], start=1.0)
# note: this vairable also includes the voltage magnitude limits and a starting value

# Add active power generation variable pg for each generator (including limits)
@variable(model, ref[:gen][i]["pmin"] <= pg[i in keys(ref[:gen])] <= ref[:gen][i]["pmax"])
# Add reactive power generation variable qg for each generator (including limits)
@variable(model, ref[:gen][i]["qmin"] <= qg[i in keys(ref[:gen])] <= ref[:gen][i]["qmax"])

# Add power flow variables p to represent the active power flow for each branch
@variable(model, -ref[:branch][l]["rate_a"] <= p[(l,i,j) in ref[:arcs]] <= ref[:branch][l]["rate_a"])
# Add power flow variables q to represent the reactive power flow for each branch
@variable(model, -ref[:branch][l]["rate_a"] <= q[(l,i,j) in ref[:arcs]] <= ref[:branch][l]["rate_a"])
# note: ref[:arcs] includes both the from (i,j) and the to (j,i) sides of a branch
@variable(model, qres[i in keys(ref[:gen])])

# Add Objective Function
# ----------------------

# index representing which side the HVDC line is starting

# Minimize the cost of active power generation and cost of HVDC line usage
# assumes costs are given as quadratic functions
b1 = [0.86, 0.68*2,0.75*2,0.60*2,0.75*2,0.73*2]   # bids from all generators
b2 = [0.46, 0.39*5, 0.43*5,0.5*5,0.9*5,0.38*5]    # bids from all generators
b1[1] = b1[1] * b1_1
b2[1] = b2[1] * b2_1
R =[0.047,0.047,0.047,0.047,0.101,0.101]
d=[0.684511335,0.64412269,0.61306915,0.599733283,0.588874071,0.59801867,0.626786054,0.651743189,0.706039246,0.787007049,0.839016956,0.852733854,0.870642027,0.834254144,0.816536483,0.81939417,0.874071252,1,0.983615927,0.936368832,0.887597638,0.809297009,0.745856354,0.733473042484283,0.784511335,0.74412269,0.713069156,0.699733283,0.688874071,0.69801867,0.726786054,0.751743189,0.806039246,0.837007049,0.889016956,0.902733854,0.920642027,0.884254144,0.866536483,0.86939417,0.924071252,0.95,1,0.986368832,0.937597638,0.859297009,0.745856354,0.8334730421,0.854511335,0.81412269,0.783069156,0.769733283,0.758874071,0.76801867,0.796786054,0.821743189,0.876039246,0.907007049,0.959016956,0.972733854,0.990642027,0.954254144,0.936536483,0.93939417,0.994071252,1,1,1,1,0.929297009,0.815856354,0.903473042,0.754511335,0.71412269,0.683069156,0.669733283,0.658874071,0.66801867,0.696786054,0.721743189,0.776039246,0.807007049,0.859016956,0.872733854,0.890642027,0.854254144,0.836536483,0.83939417,0.894071252,0.9,0.9,1,0.989,0.829297009,0.715856354,0.803473042,0.819511335,0.77912269,0.748069156,0.734733283,0.723874071,0.73301867,0.761786054,0.786743189,0.841039246,0.872007049,0.924016956,0.937733854,0.955642027,0.919254144,0.901536483,0.90439417,0.959071252,0.965,1,0.989,0.976,0.929297009,0.815856354,0.903473042,0.799511335,0.75912269,0.728069156,0.714733283,0.703874071,0.71301867,0.741786054,0.766743189,0.821039246,0.852007049,0.904016956,0.917733854,0.935642027,0.899254144,0.881536483,0.88439417,0.939071252,0.945,0.98,0.969,0.956,0.909297009,0.795856354,0.883473042,0.759511335,0.71912269,0.688069156,0.674733283,0.663874071,0.67301867,0.701786054,0.726743189,0.781039246,0.812007049,0.864016956,0.877733854,0.895642027,0.909254144,0.891536483,0.89439417,0.949071252,0.955,1,0.979,0.966,0.869297009,0.755856354,0.843473042,0.789511335,0.74912269,0.718069156,0.704733283,0.693874071,0.70301867,0.731786054,0.756743189,0.811039246,0.842007049,0.894016956,0.907733854,0.925642027,0.939254144,0.921536483,0.92439417,0.979071252,0.985,1,1,0.996,0.899297009,0.785856354,0.873473042,0.719511335,0.67912269,0.648069156,0.634733283,0.623874071,0.63301867,0.661786054,0.686743189,0.741039246,0.772007049,0.824016956,0.837733854,0.855642027,0.869254144,0.851536483,0.85439417,0.909071252,0.915,0.93,0.93,0.926,0.829297009,0.715856354,0.803473042,0.739511335,0.69912269,0.668069156,0.654733283,0.643874071,0.65301867,0.681786054,0.706743189,0.761039246,0.842007049,0.894016956,0.907733854,0.925642027,0.939254144,0.921536483,0.92439417,0.979071252,0.985,1,1,0.996,0.899297009,0.785856354,0.873473042,0.709511335,0.66912269,0.638069156,0.624733283,0.613874071,0.62301867,0.651786054,0.676743189,0.731039246,0.812007049,0.864016956,0.877733854,0.895642027,0.909254144,0.891536483,0.89439417,0.949071252,0.955,0.97,0.97,0.966,0.869297009,0.755856354,0.843473042,0.719511335,0.67912269,0.648069156,0.634733283,0.623874071,0.63301867,0.661786054,0.686743189,0.781039246,0.862007049,0.914016956,0.927733854,0.945642027,0.939254144,0.921536483,0.92439417,0.979071252,0.985,1,1,0.996,0.899297009,0.785856354,0.873473042,0.669511335,0.62912269,0.598069156,0.584733283,0.573874071,0.58301867,0.611786054,0.636743189,0.731039246,0.812007049,0.934016956,0.947733854,0.965642027,0.959254144,0.941536483,0.94439417,1,1,1,1,1,0.919297009,0.805856354,0.893473042,0.719511335,0.67912269,0.648069156,0.634733283,0.623874071,0.63301867,0.661786054,0.686743189,0.741039246,0.772007049,0.824016956,0.837733854,0.855642027,0.869254144,0.851536483,0.85439417,0.909071252,0.915,0.93,0.93,0.926,0.829297009,0.715856354,0.803473042,0.739511335,0.69912269,0.668069156,0.654733283,0.643874071,0.65301867,0.681786054,0.706743189,0.761039246,0.842007049,0.894016956,0.907733854,0.925642027,0.939254144,0.921536483,0.92439417,0.979071252,0.985,1,1,0.996,0.899297009,0.785856354,0.873473042,0.719511335,0.67912269,0.648069156,0.634733283,0.623874071,0.63301867,0.661786054,0.686743189,0.741039246,0.772007049,0.824016956,0.837733854,0.855642027,0.869254144,0.851536483,0.85439417,0.909071252,1,1,1,0.956,0.829297009,0.715856354,0.803473042,0.739511335,0.69912269,0.668069156,0.654733283,0.643874071,0.65301867,0.681786054,0.706743189,0.761039246,0.792007049,0.844016956,0.857733854,0.875642027,0.889254144,0.871536483,0.87439417,0.929071252,0.98,1,0.98,0.976,0.849297009,0.735856354,0.823473042,0.639511335,0.59912269,0.568069156,0.554733283,0.543874071,0.55301867,0.581786054,0.606743189,0.661039246,0.692007049,0.744016956,0.757733854,0.775642027,0.789254144,0.771536483,0.77439417,0.829071252,0.929071252,0.959071252,0.989071252,0.771536483,0.77439417,0.829071252,0.859071252,0.719511335,0.67912269,0.648069156,0.634733283,0.623874071,0.63301867,0.661786054,0.686743189,0.741039246,0.772007049,0.824016956,0.837733854,0.855642027,0.869254144,0.85153648,0.85439417,0.909071252,1,1,0.976,0.851536483,0.85439417,0.909071252,0.939071252,0.749511335,0.70912269,0.678069156,0.664733283,0.653874071,0.66301867,0.691786054,0.716743189,0.771039246,0.802007049,0.854016956,0.867733854,0.885642027,0.899254144,0.881536483,0.88439417,0.939071252,0.978,0.998,1,0.881536483,0.88439417,0.939071252,0.969071252,0.709511335,0.66912269,0.638069156,0.624733283,0.613874071,0.62301867,0.651786054,0.676743189,0.731039246,0.762007049,0.814016956,0.827733854,0.845642027,0.859254144,0.841536483,0.84439417,0.899071252,0.938,0.958,0.96,0.841536483,0.84439417,0.899071252,0.929071252,0.689511335,0.64912269,0.618069156,0.604733283,0.593874071,0.60301867,0.631786054,0.656743189,0.711039246,0.742007049,0.794016956,0.807733854,0.825642027,0.839254144,0.821536483,0.84439417,0.899071252,0.938,0.958,0.96,0.841536483,0.84439417,0.899071252,0.929071252,0.719511335,0.67912269,0.648069156,0.634733283,0.623874071,0.63301867,0.661786054,0.686743189,0.741039246,0.772007049,0.824016956,0.837733854,0.855642027,0.869254144,0.851536483,0.87439417,0.929071252,0.968,1,0.99,0.871536483,0.87439417,0.929071252,0.959071252,0.739511335,0.69912269,0.668069156,0.654733283,0.643874071,0.65301867,0.681786054,0.706743189,0.761039246,0.792007049,0.844016956,0.857733854,0.875642027,0.889254144,0.871536483,0.89439417,0.949071252,0.988,1,1,0.891536483,0.89439417,0.949071252,0.979071252,0.669511335,0.62912269,0.598069156,0.584733283,0.573874071,0.58301867,0.611786054,0.636743189,0.691039246,0.722007049,0.774016956,0.787733854,0.805642027,0.819254144,0.801536483,0.82439417,0.949071252,0.988,1,1,0.891536483,0.89439417,0.949071252,0.979071252,0.699511335,0.65912269,0.628069156,0.614733283,0.603874071,0.61301867,0.641786054,0.666743189,0.721039246,0.752007049,0.804016956,0.817733854,0.835642027,0.849254144,0.831536483,0.85439417,0.979071252,0.998,1,0.987,0.921536483,0.92439417,0.979071252,1.009071252,0.709511335,0.66912269,0.638069156,0.624733283,0.613874071,0.62301867,0.651786054,0.676743189,0.731039246,0.762007049,0.814016956,0.827733854,0.845642027,0.859254144,0.841536483,0.86439417,1,0.998,1,0.987,0.921536483,0.92439417,0.979071252,1.009071252,0.739511335,0.69912269,0.668069156,0.654733283,0.643874071,0.65301867,0.681786054,0.706743189,0.761039246,0.792007049,0.844016956,0.857733854,0.875642027,0.889254144,0.871536483,0.89439417,1.03,1.028,1.03,1.017,0.951536483,0.95439417,1.009071252,1.039071252,0.739511335,0.69912269,0.668069156,0.654733283,0.643874071,0.65301867,0.681786054,0.706743189,0.761039246,0.792007049,0.844016956,0.857733854,0.875642027,0.889254144,0.871536483,0.87439417,0.929071252,0.98,1,0.98,0.976,0.849297009,0.735856354,0.823473042,0.709511335,0.66912269,0.638069156,0.624733283,0.613874071,0.62301867,0.651786054,0.676743189,0.731039246,0.762007049,0.814016956,0.827733854,0.845642027,0.859254144,0.841536483,0.84439417,0.899071252,0.938,0.958,0.96,0.841536483,0.84439417,0.899071252,0.929071252]

@variable(model, price[i in keys(ref[:gen])])
@variable(model, qbb[i in keys(ref[:gen])] ==ref[:gen][i]["qmax"]*0.1 )
@variable(model, qmm[i in keys(ref[:gen])])
@objective(model, Min, sum((price[i]*qg[i])+(b2[i]*d[t]*qres[i]*qres[i])+(b1[i]*d[t]*qmm[i]*R[i]) for (i,gen) in ref[:gen])-(b2[1]*d[t]*qres[1]*qres[1])+(b1[1]*d[t]*qmm[1]*R[1])+(b2[1]*qres[1]*qres[1])+(b1[1]*qmm[1]*R[1]))

c1 = [0.86, 0.68,0.75,0.60,0.75,0.73]   #actual cost
c2 = [0.46, 0.39, 0.43,0.5,0.9,0.38]    #actual cost
bij = [21.92, 33.104, 7.142, 62.222, 8.703, 4.404]

# Add Constraints
# 0
for (i,gen) in ref[:gen]
    @constraint(model, (price[i] -(c1[i]*qg[i]*2)-(c2[i]))==0)
end
# Fix the voltage angle to zero at the reference bus
for (i,bus) in ref[:ref_buses]
    @constraint(model, va[i] == 0)
end

for (i,gen) in ref[:gen]
    @constraint(model, qres[i] == ref[:gen][i]["qmax"] -qg[i])
end

for (i,gen) in ref[:gen]
    @constraint(model, qmm[i] <= (2-(vm[i]*vm[i]))*bij[i])
end

for (i,gen) in ref[:gen]
    @constraint(model,price[i] <= b1[i]+(2*b2[i]*(qg[i]-qbb[i]*0.5/100)))
end
#for (i,gen) in ref[:gen]
#    @constraint(model, qmm[i] <= (2-(vm[i]*vm[i]))*b[i])
#end
# Nodal power balance constraints
for (i,bus) in ref[:bus]
    # Build a list of the loads and shunt elements connected to the bus i
    bus_loads = [ref[:load][l] for l in ref[:bus_loads][i]]
    bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]

    # Active power balance at node i
    @constraint(model,
        sum(p[a] for a in ref[:bus_arcs][i]) +                  # sum of active power flow on lines from bus i +
        sum(p_dc[a_dc] for a_dc in ref[:bus_arcs_dc][i]) ==     # sum of active power flow on HVDC lines from bus i =
        sum(pg[g] for g in ref[:bus_gens][i]) -                 # sum of active power generation at bus i -
        sum(load["pd"]*d[t] for load in bus_loads) -                 # sum of active load consumption at bus i -
        sum(shunt["gs"] for shunt in bus_shunts)*vm[i]^2        # sum of active shunt element injections at bus i
    )

    # Reactive power balance at node i
    @constraint(model,
        sum(q[a] for a in ref[:bus_arcs][i]) +                  # sum of reactive power flow on lines from bus i +
        sum(q_dc[a_dc] for a_dc in ref[:bus_arcs_dc][i]) ==     # sum of reactive power flow on HVDC lines from bus i =
        sum(qg[g] for g in ref[:bus_gens][i]) -                 # sum of reactive power generation at bus i -
        sum(load["qd"]*d[t] for load in bus_loads) +                 # sum of reactive load consumption at bus i -
        sum(shunt["bs"] for shunt in bus_shunts)*vm[i]^2        # sum of reactive shunt element injections at bus i
    )
end

# Branch power flow physics and limit constraints
for (i,branch) in ref[:branch]
    # Build the from variable id of the i-th branch, which is a tuple given by (branch id, from bus, to bus)
    f_idx = (i, branch["f_bus"], branch["t_bus"])
    # Build the to variable id of the i-th branch, which is a tuple given by (branch id, to bus, from bus)
    t_idx = (i, branch["t_bus"], branch["f_bus"])
    # note: it is necessary to distinguish between the from and to sides of a branch due to power losses

    p_fr = p[f_idx]                     # p_fr is a reference to the optimization variable p[f_idx]
    q_fr = q[f_idx]                     # q_fr is a reference to the optimization variable q[f_idx]
    p_to = p[t_idx]                     # p_to is a reference to the optimization variable p[t_idx]
    q_to = q[t_idx]                     # q_to is a reference to the optimization variable q[t_idx]
    # note: adding constraints to p_fr is equivalent to adding constraints to p[f_idx], and so on

    vm_fr = vm[branch["f_bus"]]         # vm_fr is a reference to the optimization variable vm on the from side of the branch
    vm_to = vm[branch["t_bus"]]         # vm_to is a reference to the optimization variable vm on the to side of the branch
    va_fr = va[branch["f_bus"]]         # va_fr is a reference to the optimization variable va on the from side of the branch
    va_to = va[branch["t_bus"]]         # va_fr is a reference to the optimization variable va on the to side of the branch

    # Compute the branch parameters and transformer ratios from the data
    g, b = PowerModels.calc_branch_y(branch)
    tr, ti = PowerModels.calc_branch_t(branch)
    g_fr = branch["g_fr"]
    b_fr = branch["b_fr"]
    g_to = branch["g_to"]
    b_to = branch["b_to"]
    tm = branch["tap"]^2
    # note: tap is assumed to be 1.0 on non-transformer branches


    # AC Power Flow Constraints

    # From side of the branch flow
    @NLconstraint(model, p_fr ==  (g+g_fr)/tm*vm_fr^2 + (-g*tr+b*ti)/tm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-b*tr-g*ti)/tm*(vm_fr*vm_to*sin(va_fr-va_to)) )
    @NLconstraint(model, q_fr == -(b+b_fr)/tm*vm_fr^2 - (-b*tr-g*ti)/tm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-g*tr+b*ti)/tm*(vm_fr*vm_to*sin(va_fr-va_to)) )

    # To side of the branch flow
    @NLconstraint(model, p_to ==  (g+g_to)*vm_to^2 + (-g*tr-b*ti)/tm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-b*tr+g*ti)/tm*(vm_to*vm_fr*sin(va_to-va_fr)) )
    @NLconstraint(model, q_to == -(b+b_to)*vm_to^2 - (-b*tr+g*ti)/tm*(vm_to*vm_fr*cos(va_fr-va_to)) + (-g*tr-b*ti)/tm*(vm_to*vm_fr*sin(va_to-va_fr)) )

    # Voltage angle difference limit
    @constraint(model, va_fr - va_to <= branch["angmax"])
    @constraint(model, va_fr - va_to >= branch["angmin"])

    # Apparent power limit, from side and to side
    @constraint(model, p_fr^2 + q_fr^2 <= branch["rate_a"]^2)
    @constraint(model, p_to^2 + q_to^2 <= branch["rate_a"]^2)
end


###############################################################################
# 3. Solve the Optimal Power Flow Model and Review the Results
###############################################################################

# Solve the optimization problem
optimize!(model)

# Check that the solver terminated without an error
#println("The solver termination status is $(termination_status(model))")

# Check the value of the objective function

cost = objective_value(model)
#println("The cost of generation is $(cost).")

# Check the value of an optimization variable
# Example: Active power generated at generator 1
#qg1 = value(qg[1])
#price1 = value(price[1])
#println("The reactive power generated at generator 1 is $(qg1*ref[:baseMVA]) MVAr.")
#println(JuMP.value.(price))
#println(JuMP.value.(qg*ref[:baseMVA]))

# note: the optimization model is in per unit, so the baseMVA value is used to restore the physical units
qty = JuMP.value.(qg*ref[:baseMVA])[1]
rewards = [(value(price[i])*value(qg[i]) - c1[i]*(value(qg[i])-0.075) - (c2[i]* (value(qg[i])-0.075) * (value(qg[i])-0.075))) for i in range(1,stop=6)]
qty = [value(qg[i]) for i in range(1,stop=6)]
rewards, qty
end