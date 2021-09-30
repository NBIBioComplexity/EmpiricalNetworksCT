import os
import sys
import shutil
import itertools
import gc
import json
import csv
import operator
import networkx as nx
from networkx.algorithms import community #This part of networkx, for community detection, needs to be imported separately.
import pandas as pd
import numpy as np
import scipy.stats as ss
from datetime import datetime
import random
import time
import pickle
import glob


num_days = 49
dt = 300

timechunks = pickle.load( open('btdata_timechunks_symmetrized_85dBm.pkl', "rb" ) )

n_users = 643 # Note, this is hardcoded for now but can easily be inferred from the data.



# Helper functions for parameter assignment:
def str2list(string):
    if bool(string[0] == "[" and string[-1] == "]" and
            "--" not in string):
        # Avoid parsing line specification as list.
        li = string[1:-1].split(",")
        for i in range(len(li)):
            li[i] = str2list(li[i])
        return li
    else:
        return parseval(string)


def parseval(value):
    try:
        value = json.loads(value)
    except ValueError:
        # json expects true/false, not True/False
        if value in ["True", "False"]:
            value = eval(value)
        elif "True" in value or "False" in value:
            value = eval(value)

    if isinstance(value, dict):
        value = convert(value)
    elif isinstance(value, list):
        value = convert(value)
    return value

def parse_command_line():
    cmd_kwargs = dict()
    for s in sys.argv[1:]:
        if s in ["-h", "--help", "help"]:
            key, value = "help", "true"
        elif s.count('=') == 0:
            key, value = s, "true"
        elif s.count('=') == 1:
            key, value = s.split('=', 1)
        else:
            raise TypeError(
                "Only kwargs separated with at the most a single '=' allowed.")

        value = parseval(value)
        if isinstance(value, str):
            value = str2list(value)

        cmd_kwargs[key] = value
    return cmd_kwargs

# Set parameters of simulation:
cmdparameters = dict(
    n_sims = 5,
    rtest_ratio = 0,
    ct_threshold = 250,
    retention_days=0,
    rand_scheme = "true",
    directory = "CT_outdata",
    p_inf = 8/1000,
    Qdays=5,
)

cmd_kwargs = parse_command_line()
cmdparameters.update(**cmd_kwargs)

def print_available_params(pars):
    print("The current parameter values are:")
    for p in pars:
        print(f"{p} = {pars[p]}")
    print("Values can be set as command line arguments with syntax param=value.")

print_available_params(cmdparameters)

def timechunks_to_daychunks(timechunks):
    print("Reformatting temporal network data ...")
    daychunks = dict()
    n = 0
    m = 0
    k = 0
    n_max = len(timechunks)-1
    while n <= n_max:
        #print(n)
        if n == 0:
            daychunks[0] = dict()
            daychunks[0][0] = timechunks[0]
        else:
            if timechunks[n-1]["day"].iloc[0] == timechunks[n]["day"].iloc[0]:
                k += 1
            else:
                m += 1
                daychunks[m] = dict()
                k = 0
            daychunks[m][k] = timechunks[n]
        n += 1
        if n % round(n_max/5) == 0:
            print("Progress: ", round(n*100/n_max), '%')
            print("Step", n, "out of", n_max)
    return daychunks

def shuffle_days(daychunk, shuffles, preserve_weekends=False):
    shuffled = 0
    max_idx = len(daychunk)-1
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    weekend = ["Saturday", "Sunday"]
    if preserve_weekends:
        while shuffled <= shuffles:
            from_idx = random.randint(0, max_idx)
            to_idx = random.randint(0, max_idx)
            from_day = daychunk[from_idx][0].iloc[0]["day"]
            if from_day in weekend:
                to_day = daychunk[to_idx][0].iloc[0]["day"]
                while to_day in weekdays:
                    to_idx = random.randint(0, max_idx)
                    to_day = daychunk[to_idx][0].iloc[0]["day"]
                #print("Swapping", from_day, "with", to_day)
            if from_day in weekdays:
                to_day = daychunk[to_idx][0].iloc[0]["day"]
                while to_day in weekend:
                    to_idx = random.randint(0, max_idx)
                    to_day = daychunk[to_idx][0].iloc[0]["day"]
                #print("Swapping", from_day, "with", to_day)
            # Must I use .copy() or some such?
            from_tmp = daychunk[from_idx]
            to_tmp = daychunk[to_idx]
            daychunk[from_idx] = to_tmp
            daychunk[to_idx] = from_tmp
            shuffled += 1
    else:
        while shuffled <= shuffles:
            from_idx = random.randint(0, max_idx)
            to_idx = random.randint(0, max_idx)
            # Must I use .copy() or some such?
            from_tmp = daychunk[from_idx]
            to_tmp = daychunk[to_idx]
            daychunk[from_idx] = to_tmp
            daychunk[to_idx] = from_tmp
            shuffled += 1
    return daychunk

dc = timechunks_to_daychunks(timechunks)

# Function to check if infection is possible:
def infection_possible(I_u,Pu,S_su,Q_u,Q_su):
    inf_pos = False
    if I_u or Pu:
        if S_su:
            if not Q_u:
                if not Q_su:
                    inf_pos = True
    if I_u or Pu:
        #print("Incoming state state, (I_u,Pu,S_su,Q_u,Q_su)=", (I_u,Pu,S_su,Q_u,Q_su))
        #print("Infection possible?", inf_pos)
        pass
    return inf_pos

def tests_positive(Iu,Pu):
    pos = False
    #if Iu or Pu: # Presymptomatics can be detected
    #    pos = True
    if Iu: # Presymptomatics cannot be detected.
        pos = True
    return pos

def propagate_time(df, S, E, I, R, Q, infectree, R0tree, contactree, userinfo, quarantine_reasons, Erates, tstep, parameters):
    t = tstep;
    tests = 0

    G = nx.from_pandas_edgelist(df, source='user', target='scanned_user')

    if parameters["randomized"] == "edgeswap":
        n_swap = 4*G.number_of_edges() 
        max_try = 10 * n_swap
        try:
            G = nx.double_edge_swap(G, nswap=n_swap, max_tries=max_try)
        except:
            print("Graph not edgeswapped ... ")
        edges = G.edges
    elif parameters["randomized"] == "randomize":
        n_edges = len(G.edges)
        edges = []
        edges_added = 0
        while edges_added < n_edges:
            edge_a = random.randint(1,n_users)
            edge_b = random.randint(1,n_users)
            edgepair = (edge_a, edge_b)
            while edgepair in edges or edge_a == edge_b: # Draw new edges
                edge_a = random.randint(1,n_users)
                edge_b = random.randint(1,n_users)
                edgepair = (edge_a, edge_b)
            edges.append(edgepair)
            edges_added += 1
    else:
        edges = G.edges
    p_inf = parameters["p_inf"]
    I_to_R_rate = parameters["I_to_R_rate"]
    r_test = parameters["rtest_ratio"] * I_to_R_rate
    Qtime = parameters["Qtime"]
    contact_threshold = parameters["ct_threshold"]
    C = np.zeros(*I.shape)
    # Testing and quarantining routine:
    for u in range(1,parameters["n_users"]+1):
        if userinfo[u]["known_positive"]:
            pass
        else:
            if tests_positive(I[u-1], E[len(E)][u-1]):
                if random.random() <= r_test:
                    tests += 1
                    userinfo[u]["known_positive"] = True
                    Q[u-1] = 1
                    quarantine_reasons["randtest"] += 1
                    userinfo[u]["quarantine_time"] = 0
                    userinfo[u]["quarantine_timestamp"] = t
                    userinfo[u]["been_quarantined"] = True
                    # Now go through the contacts of the newly found positive:
                    known_contacts = dict()
                    for contact in contactree[u]:
                        if contact[0] in known_contacts: # The contact has been met before
                            known_contacts[contact[0]]["encounters"] += 1
                            if contact[1] > known_contacts[contact[0]]["meeting_timestamp"]:
                                known_contacts[contact[0]]["meeting_timestamp"] = contact[1]
                        else:
                            known_contacts[contact[0]] = dict()
                            known_contacts[contact[0]]["encounters"] = 1
                            known_contacts[contact[0]]["meeting_timestamp"] = contact[1]
                    for contact in known_contacts:
                        if known_contacts[contact]["encounters"] >= contact_threshold and (not Q[contact-1]) and (not userinfo[contact]["known_positive"]):
                            # Put contact in quarantine, if he hasn't already been put in one AFTER meeting this person
                            to_quarantine = False
                            t_meet = known_contacts[contact]["meeting_timestamp"]
                            t_quar = userinfo[contact]["quarantine_timestamp"]
                            if t_meet > t_quar: # If the meeting happened later than the last quarantining
                                to_quarantine = True
                            if to_quarantine:
                                userinfo[contact]["quarantine_time"] = 0
                                userinfo[contact]["quarantine_timestamp"] = t
                                userinfo[contact]["been_quarantined"] = True
                                Q[contact-1] = 1
                                quarantine_reasons["CT"] += 1

    # Disease spreading routine:
    edge_arr = np.array(edges, dtype=int)
    users = edge_arr[:,0]-1
    susers = edge_arr[:,1]-1
    C[users] += 1
    C[susers] += 1
    for e in edges:
        u = e[0]-1
        su = e[1]-1
        su_exp__by_u = random.random() < p_inf
        u_exp__by_su = random.random() < p_inf
        if infection_possible(I[u],E[len(E)][u],S[su],Q[u],Q[su]):
            S[su] -= su_exp__by_u
            E[1][su] += su_exp__by_u
            if su_exp__by_u:
                infectree[e[0]]["users_infected"].append(e[1])
                infectree[e[1]]["t_infected"] = tstep
        elif infection_possible(I[su],E[len(E)][su],S[u],Q[su],Q[u]):
            S[u] -= u_exp__by_su
            E[1][u] += u_exp__by_su
            if u_exp__by_su:
                infectree[e[1]]["users_infected"].append(e[0])
                infectree[e[0]]["t_infected"] = tstep
        # Add (contact,timestep) pairs to contact-trees.
        su_t_pair = (e[1],tstep)
        u_t_pair = (e[0],tstep)
        if (not Q[u]) and (not Q[su]):
            contactree[e[0]].append(su_t_pair)  # Add (or re-add!) contact
            contactree[e[1]].append(u_t_pair) # Add (or re-add!) contact
        if I[u]:
            if su_exp__by_u:
                if e[1] in R0tree[e[0]]:
                    pass
                else:
                    R0tree[e[0]].append(e[1])
        if I[su]:
            if u_exp__by_su:
                if e[0] in R0tree[e[1]]:
                    pass
                else:
                    R0tree[e[1]].append(e[0])
    for idx in range(parameters["n_users"]):
        # Quarantine remove routine:
        u = idx+1
        if Q[idx]:
            #print("Q_in =", Q[idx])
            if userinfo[u]["quarantine_time"] >= Qtime: # Minimum quarantine time has elapsed
                tests += 1 # We test before (potential) release
                if tests_positive(I[idx], E[len(E)][idx]):
                    # Reset quarantine time:
                    userinfo[u]["quarantine_time"] = 0
                    userinfo[u]["quarantine_timestamp"] = t
                    userinfo[u]["been_quarantined"] = True
                    userinfo[u]["known_positive"] = True
                    # Now go through the contacts of the newly found positive:
                    known_contacts = dict()
                    for contact in contactree[u]:
                        if contact[0] in known_contacts: # The contact has been met before
                            known_contacts[contact[0]]["encounters"] += 1
                            if contact[1] > known_contacts[contact[0]]["meeting_timestamp"]:
                                known_contacts[contact[0]]["meeting_timestamp"] = contact[1]
                        else:
                            known_contacts[contact[0]] = dict()
                            known_contacts[contact[0]]["encounters"] = 1
                            known_contacts[contact[0]]["meeting_timestamp"] = contact[1]
                        for contact in known_contacts:
                            if known_contacts[contact]["encounters"] >= contact_threshold and (not Q[contact-1]) and (not userinfo[contact]["known_positive"]):
                                # Put contact in quarantine, if he hasn't already been put in one AFTER meeting this person
                                to_quarantine = False
                                t_meet = known_contacts[contact]["meeting_timestamp"]
                                t_quar = userinfo[contact]["quarantine_timestamp"]
                                if t_meet > t_quar: # If the meeting happened later than the last quarantining
                                    to_quarantine = True
                                if to_quarantine:
                                    userinfo[contact]["quarantine_time"] = 0
                                    userinfo[contact]["quarantine_timestamp"] = t
                                    userinfo[contact]["been_quarantined"] = True
                                    Q[contact-1] = 1
                                    quarantine_reasons["CT"] += 1
                else: # The person tests negative and will be released
                    userinfo[u]["quarantine_time"] = -1
                    Q[idx] = 0
            else: # Person is not through with quarantine
                userinfo[u]["quarantine_time"] += 1
    # Transition though the E stages (and finally into I):
    i = 1
    while i < len(E):
        E_to_next = E[i] * (np.random.rand(*I.shape) < Erates[i]).astype(int)
        E[i] -= E_to_next
        E[i+1] += E_to_next
        i += 1
    # Now i = len(E), so we go from E[i] -> I
    E_to_next = E[i] * (np.random.rand(*I.shape) < Erates[i]).astype(int)
    E[i] -= E_to_next
    I = I + E_to_next
    # Delete all of infected person's contacts that are older than
    # n timesteps, at the time of being exposed.
    carried = 0
    notcarried = 0
    for idx in range(parameters["n_users"]):
        if E_to_next[idx]:
            u = idx+1
            # This person just went into I-state, and thus we delete all the
            # contacts deemed irrelevant.
            new_contactree = []
            for contact in contactree[u]:
                if tstep-contact[1] <= parameters["retention_time"]:
                    new_contactree.append(contact)
                    carried += 1
                else:
                    notcarried += 1
            contactree[u] = new_contactree
    # Some go from I to R:
    I_to_R = I * (np.random.rand(*I.shape) < I_to_R_rate).astype(int)
    I = I - I_to_R
    R = R + I_to_R
    return S, E, I, R, C, Q, tests, infectree, R0tree, contactree, userinfo, quarantine_reasons

def init_arrs(daychunks, n_users, p_inf0, days, parameters, shuffledays=True, preserve_weekends=True):
    # Establish a local (possibly shuffled) daychunks:
    outdays = dict()
    if not shuffledays:
        day = 0
        while day < days:
            outdays[day] = daychunks[day % len(daychunks)]
            day += 1
    else:
        day = 0
        shuffles = 2 * len(daychunks) # Should be high enough that edges are rarely left unshuffled
        days_added_cur = 0
        while day < days:
            if days_added_cur == 0:
                daychunks = shuffle_days(daychunks, shuffles, preserve_weekends)
            outdays[day] = daychunks[days_added_cur]
            days_added_cur += 1
            if days_added_cur == len(daychunks):
                days_added_cur = 0
            day += 1
    # Count number of timesteps in a primitive way:
    n = 0
    day = 0
    while day < len(outdays):
        dn = len(outdays[day])
        n += dn
        day += 1
    print("Number of timesteps: ", n)
    # Initiate matrices for holding all timeseries:
    S_t = np.zeros(n, int)
    E_t = np.zeros(n, int)
    I_t = np.zeros(n, int)
    R_t = np.zeros(n, int)
    C_t = np.zeros(n, int)
    Q_t = np.zeros(n, int)
    tests_t = np.zeros(n, int)
    # At first, let everyone be S (susceptible):
    S = np.ones((n_users), int)
    #E = np.zeros((n_users), int)
    # Dictionary to hold the stages of the exposed state:
    E = {1: np.zeros((n_users), int), 2: np.zeros((n_users), int), 3: np.zeros((n_users), int)}
    # Dictionary to hold the rates to leave the stages of the exposed state:
    Erates = {1: 2.9/1000, 2: 2.9/1000, 3: 2.9/1000}
    I = np.zeros((n_users), int)
    R = np.zeros((n_users), int)
    Q = np.zeros((n_users), int)
    # Then remove a few people and put them in the I state:
    S -= R
    I = ((np.random.rand(n_users)+R) < p_inf0).astype(int) # Never true for those where R=1
    S -= I
    infectree = dict()
    R0tree = dict()
    contactree = dict()
    userinfo = dict()
    for u in range(1,n_users+1):
        infectree[u] = dict()
        if I[u-1] == 1:
            infectree[u]["t_infected"] = 0
        else:
            infectree[u]["t_infected"] = -1
        infectree[u]["users_infected"] = []
        R0tree[u] = []
        contactree[u] = []
        userinfo[u] = dict()
        userinfo[u]["quarantine_time"] = -1
        userinfo[u]["quarantine_timestamp"] = -1
        userinfo[u]["known_positive"] = False
        userinfo[u]["been_quarantined"] = False
    tstep = 0
    S_t[tstep] = np.sum(S)
    E_t[tstep] = np.sum(sum(E.values()))
    I_t[tstep] = np.sum(I)
    R_t[tstep] = np.sum(R)
    return outdays, S, E, I, R, Q, Erates, S_t, E_t,  I_t, R_t, C_t, Q_t, tests_t, infectree, R0tree, contactree, userinfo, n

def main(daychunks, days, n_users, p_inf0, randomized=False, shuf_days = False, preserve_weekends=True, rtest_ratio=1, ct_threshold=1, retention_days=0, verbose=True, p_inf=5/1000, Qdays=5):
    retention_time = int(retention_days * 288)
    parameters = {
        "days" : days,
        "n_users" : n_users,
        "p_inf0" : p_inf0,
        "randomized" : randomized,
        "shuf_days" : shuf_days,
        "preserve_weekends" : preserve_weekends,
        "rtest_ratio" : rtest_ratio,
        "ct_threshold" : ct_threshold,
        "retention_days" : retention_days,
        "retention_time" : retention_time,
        "p_inf" : p_inf,
        #"I_to_R_rate" : 0.00077160494, # Corresponds to 4.5 days, since 5min/4.5day = 0.00077160494
        "I_to_R_rate" : 0.00086805556, # Corresponds to 4.0 days
        "Qtime" : 288*Qdays,
    }
    daychunks, S, E, I, R, Q, Erates, S_t, E_t, I_t, R_t, C_t, Q_t, tests_t, infectree, R0tree, contactree, userinfo, n = init_arrs(daychunks, n_users, p_inf0, days, parameters, shuf_days, preserve_weekends)
    parameters["Erates"] = Erates
    quarantine_reasons = dict()
    quarantine_reasons["randtest"] = 0
    quarantine_reasons["CT"] = 0
    quarantine_reasons["CT_of_CT"] = 0
    if verbose:
        print("Starting simulation ...")
    debug = False
    day = 0
    print('With retention_days =', retention_days, 'retention_time =', retention_time)
    tstep = 0
    while day < days:
        chunk = 0
        timechunks = daychunks[day]
        while chunk < len(timechunks):
            #print("Running tstep", tstep)
            S, E, I, R, C, Q, tests, infectree, R0tree, contactree, userinfo, quarantine_reasons = propagate_time(timechunks[chunk], S, E, I, R, Q, infectree, R0tree, contactree, userinfo, quarantine_reasons, Erates, tstep, parameters)
            S_t[tstep] = np.sum(S)
            E_t[tstep] = np.sum(sum(E.values()))
            I_t[tstep] = np.sum(I)
            R_t[tstep] = np.sum(R)
            C_t[tstep] = np.sum(C)
            Q_t[tstep] = np.sum(Q)
            tests_t[tstep] = tests
            tstep += 1
            chunk += 1
            if tstep % round(n/10) == 0:
                if verbose:
                    print("Progress: ", round(tstep*100/n), '%')
                    print("I + E =", round(100 * (np.sum(I)+np.sum(sum(E.values())))/n_users, 2), '%')
        day += 1
    print("Done.")
    return S_t, E_t, I_t, R_t, C_t, Q_t, tests_t, infectree, R0tree, contactree, quarantine_reasons, parameters

############## Run ensemble #####################
def run_ensemble(n_sims=20, rtest_ratio=1, ct_threshold=1, retention_days=0, verbose=False, save=False, randomized=False, directory=False, p_inf=5/1000, Qdays=5):
    days = 80
    p_inf0 = 0.02
    shuf_days = True
    preserve_weekends = False
    results = dict()
    if randomized == "edgeswap":
        mode = "edgeswap"
    elif randomized == "randomize":
        mode = "randomized"
    else:
        mode = "trueconnections"
    basefname = 'ensemble_' + mode + "_rtest_" + str(round(rtest_ratio,4)) + "_ctthresh_" + str(int(ct_threshold)) + '_ret_' + str(retention_days) + '_' + "pinf_"+ str(round(p_inf,5)) + '_' + str(days) + 'days'
    print("Running ensemble with identifier", '"' + basefname + '"')
    sim = 1
    while sim <= n_sims:
        result = dict()
        print("Simulation", sim, "out of", n_sims,  "started at ", datetime.now())
        tic = time.perf_counter()
        # Omitting contactree in output data:
        result["S"], result["E"], result["I"], result["R"], result["C"], result["Q"], result["tests"], result["infectree"], result["R0tree"], __, result["quarantine_reasons"], result["parameters"]  = main(dc, days, n_users, p_inf0,
                                                                               randomized=randomized,
                                                                               shuf_days=shuf_days,
                                                                               preserve_weekends=preserve_weekends,
                                                                               rtest_ratio=rtest_ratio,
                                                                               ct_threshold=ct_threshold,
                                                                               retention_days=retention_days,
                                                                               verbose=verbose,
                                                                               p_inf=p_inf,
                                                                               Qdays=Qdays)
        toc = time.perf_counter()
        print("Simulation", sim, "out of", n_sims, "ended at ", datetime.now())
        print(f"Time elapsed: {round((toc - tic),2)} s")
        results[sim] = result
        sim += 1
    print("---")
    randstr = str(random.randint(1,1e7))
    pklname = basefname + "_" + randstr + '.pkl'
    if save:
        print("Saving ensemble results to", pklname)
        if directory:
            fulldir = "" + directory
            os.system("mkdir -p " + fulldir)
            f = open(fulldir + "/" + pklname, "wb")
        else:
            f = open(pklname, "wb")
        pickle.dump(results,f)
        f.close()
    return results, basefname

__, ___ = run_ensemble(n_sims=cmdparameters["n_sims"], rtest_ratio=cmdparameters["rtest_ratio"], ct_threshold=cmdparameters["ct_threshold"], retention_days=cmdparameters["retention_days"], verbose=False, save=True, randomized=cmdparameters["rand_scheme"], directory=cmdparameters["directory"], p_inf=cmdparameters["p_inf"], Qdays=cmdparameters["Qdays"])
