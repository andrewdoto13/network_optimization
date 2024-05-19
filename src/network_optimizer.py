import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from sklearn.metrics.pairwise import haversine_distances
import random
import time
from math import radians

class NetworkOptimizer:
    """
Class that can perform provider network optimization. Implementation is a steepest-ascent, Hill-climbing local search 
optimization algorithm.

>>> optimizer = NetworkOptimizer(pool, members, adequacy_reqs)
>>> optimizer.optimize(num_rounds)
    """
    def __init__(self, 
                 pool: pd.DataFrame,
                 members: pd.DataFrame,
                 adequacy_reqs: pd.DataFrame,
                 user_objective: Optional[callable] = None,
                 network: Optional[pd.DataFrame] = None
                 ) -> None:
        """
    Initialize the optimizer. The pool, members, and adequacy requirements are required. If not passed, the network will start as blank and the optimizer
    will optimize it, guided by the objective function.
    
    :param pool: pandas DataFrame storing the pool of potential providers the network can have
    :param members: pandas DataFrame storing the members (beneficiaries) that the network serves
    :param adequacy_reqs: pandas DataFrame storing the adequacy requirements for the network
    :param objective: objective function that takes in a pandas dataframe and guides the algorithm
    :param network: pandas DataFrame storing the providers already contracted for the network, if any
        """
        
        self.initial_pool = pool.copy()
        self.adequacy_reqs = adequacy_reqs.copy()
        self.members = members.copy()
        self.user_objective = user_objective
        self.initial_network = network.copy() if network is not None else pd.DataFrame(columns=self.initial_pool.columns)
        self.distance_matrix = np.empty(0)
        self.performance_history = np.empty(0)
        self.move_tracker = []
        self.time_tracker = np.empty(0)
        self.total_optimization_rounds = 0
        self.adequacy_detail = np.empty(0)

        # create distance matrix
        member_coords_radians = self.members.loc[:, ["latitude", "longitude"]].applymap(lambda x: radians(x))

        if len(self.initial_network) > 0:
            all_providers_radians = pd.concat([
                self.initial_network[["latitude", "longitude"]], 
                self.initial_pool[["latitude", "longitude"]]]).applymap(lambda x: radians(x))
            
            all_provider_county_specialties = pd.concat([
                self.initial_network[["npi", "county", "specialty"]], 
                self.initial_pool[["npi", "county", "specialty"]]])
        else:
            all_providers_radians = self.initial_pool[["latitude", "longitude"]].applymap(lambda x: radians(x))

            all_provider_county_specialties = self.initial_pool[["npi", "county", "specialty"]]
            
        all_providers_index = pd.concat([self.initial_network.npi, self.initial_pool.npi])
            
        # multiply by Earth radius to get miles
        self.distance_matrix = (pd.DataFrame(data=(haversine_distances(member_coords_radians, all_providers_radians)  * 3959).round(1),
                                            index=self.members.member_id,
                                            columns=all_providers_index)
                                            .reset_index()
                                            .melt(id_vars="member_id", 
                                                  var_name="npi", 
                                                  value_name="distance")
                                            .merge(all_provider_county_specialties, on="npi", how="left")
                                            .merge(self.adequacy_reqs, on=["county", "specialty"], how="left")
                                            )
        
        self.pct_serving = (self.distance_matrix[["npi"]].merge(
            (self.distance_matrix.loc[self.distance_matrix.distance <= self.distance_matrix.distance_req].groupby("npi")["member_id"].count() / len(self.members))
            .reset_index()
            .rename({"member_id":"pct_serving"}, axis=1), on="npi", how="left").fillna(0).drop_duplicates())

        self.pool = pool.copy().merge(self.pct_serving, on=["npi"], how="left")

        self.best_network = network.copy().merge(self.pct_serving, on=["npi"], how="left") if network is not None else pd.DataFrame(columns=self.pool.columns)

    def adequacy(self, network: pd.DataFrame) -> float:
        """
    Calculate adequacy of a network using the adequacy requirements provided by the user. The returned value is a float that
    is a slight modification of the adequacy index score. It takes the network and the adequacy requirements and it returns the mean
    of the product of the percent of members with access and the percent of required providers for all the county/specialty combinations.

    :param network: pandas DataFrame with the network for which you want to calculate adequacy 
        """
        if len(network) == 0:
            return 0
        else:
            network_distances = (self.distance_matrix[["member_id", "npi", "distance"]]
                     .merge(network[["npi", "specialty", "county"]], on="npi", how="inner")
                     .merge(self.adequacy_reqs[["county", "specialty", "distance_req"]], on=["county", "specialty"], how="left"))

            network_distances["meets_distance"] = network_distances.distance <= network_distances.distance_req

            access_summary = self.adequacy_reqs.merge(
                network_distances
                .loc[network_distances.meets_distance == True]
                .groupby(["county", "specialty"])["member_id"].nunique()
                .reset_index()
                .rename({"member_id":"members_with_access"}, axis=1), on=["county", "specialty"], how="left"
            )
            
            access_summary["pct_members_with_access"] = access_summary.members_with_access / len(self.members)

            provider_counts = network_distances.groupby(["county", "specialty"])["npi"].nunique().reset_index().rename({"npi":"provider_count"}, axis=1)

            adequacy_detail = access_summary.merge(provider_counts, on=["county", "specialty"], how="left").fillna(0)

            adequacy_detail["pct_req_providers"] = adequacy_detail.provider_count / adequacy_detail.min_providers
            
            self.adequacy_detail = adequacy_detail

            mean_adequacy_score = round((adequacy_detail.pct_members_with_access.apply(lambda x: min(x,1)) * adequacy_detail.pct_req_providers.apply(lambda x: min(x,1))).mean(), 3)

            adequacy_county_specialties = len(adequacy_detail.loc[(adequacy_detail.pct_members_with_access * 100 >= adequacy_detail.min_access_pct)
                                                                  & (adequacy_detail.provider_count >= adequacy_detail.min_providers)])

            if adequacy_county_specialties == len(adequacy_detail):
                return 1
            else:
                return mean_adequacy_score
        
    def objective(self, network: pd.DataFrame) -> float:
        """
    Objective function that describes the goal of the optimization. Takes in a pandas DataFrame storing a provider network as input.
    It is the compass for the algorithm to optimize the network. The default is adequacy, but if the user passes in a function, it will use that instead.

    :param network: pandas DataFrame with the network for which you want to calculate performance
        """
        return self.adequacy(network) if self.user_objective is None else self.user_objective(self, network)

    def successor(self, network: pd.DataFrame, pool: pd.DataFrame) -> List[Tuple]:
        """
    Returns all possible moves as successor states, given the provided network and provider pool. This represents
    all possible changes to the network. If empty network, all successor states are simply the states with each pool provider
    added to the network. Important to note that each successor state only deals with one change of a provider, i.e. the smallest
    possible "step" you can take.

    :param network: pandas DataFrame with the network in its current state
    :param pool: pandas DataFrame with the pool of potential providers
        """
        # get additions
        additions = [("addition", idx) for idx in pool.sort_values(by="pct_serving", ascending=False).index.to_list()]
        if len(network) == 0:
            removals = [(None, None) for i in range(len(additions))]
            swaps = [(None, None, None) for i in range(len(additions))]
            return list(zip(additions, removals, swaps))
        else:
            # get replacement combinations within specialty
            swapDF = network.reset_index().merge(pool.reset_index(), on = ["county", "specialty"], how = "inner", suffixes= ["_network", "_pool"]).dropna()
            swapDF["pct_diff"] = swapDF.pct_serving_pool - swapDF.pct_serving_network
            swaps = [("swap",i,j) for i,j in swapDF.sort_values(by="pct_diff", ascending=False)[["index_network", "index_pool"]].values]
            # get removals
            removals =  [("removal", idx) for idx in network.sort_values(by="pct_serving", ascending=True).index.astype(int).to_list()]
            return list(zip(additions, removals, swaps))
    
    def create_state(self, network: pd.DataFrame, pool: pd.DataFrame, change: Tuple):
        """
    Create a network based on the pool and the change. This function takes the network passed in and creates a copy, then 
    makes the change that is described by the change parameter, which comes as a tuple. The first element of the tuple
    tells you what kind of change: addition, removal, or a swap. The rest of the tuple tells you the index of the row that is associated 
    with the change.

    :param network: pandas DataFrame with the network in its current state
    :param pool: pandas DataFrame with the pool of potential providers
    :param change: tuple that describes the change to be made to the network
        """
        # make a true copy of the network, not just a reference
        # see here: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.copy.html
        new_network = network.copy()

        if change[0] == "addition":
            new_network.loc[len(new_network)] = pool.loc[change[1]]
            return new_network
        elif change[0] == "removal":
            new_network = new_network.drop(change[1])
            return new_network
        else:
            new_network.loc[change[1]] = pool.loc[change[2]]
            return new_network
    
    def optimize(self, num_rounds: int) -> None:
        """
    Perform steepest-ascent local search optimization for a number of rounds input by the user.
    The algorithm takes the network, determines all the possible moves from the successor function. Then
    it calculates the performance of all the successor states and stores the best one. Repeat for number of
    rounds or until goal state is met.

    :param num_rounds: max number of rounds to perform optimization
        """
        # save performance of initial network
        if len(self.performance_history) == 0:
            self.performance_history = np.append(self.performance_history, self.objective(self.initial_network))

        # for set number of rounds
        for optim_round in range(num_rounds):
            self.total_optimization_rounds += 1
            start = time.perf_counter()
            print(f"Optimization round {self.total_optimization_rounds} ...")
            # get state changes
            state_changes = self.successor(self.best_network, self.pool)
            if len(state_changes) == 0:
                print("Pool has been exhaused")
                break
            
            for options in state_changes:
                # generate new states for the options and get the scores
                new_states = [self.create_state(self.best_network, self.pool, option) for option in options if option[0] in ["addition", "removal", "swap"]]
                new_scores = np.array([self.objective(state) for state in new_states])

                # get the best performing state and score
                best_score = np.max(new_scores)
                best_state_idx = np.argmax(new_scores)

                best_move = options[best_state_idx]
                best_state = new_states[best_state_idx]

                #if the best new move is better than the latest best performance
                if best_score > self.performance_history[-1]:
                    # store best performance and move
                    self.performance_history = np.append(self.performance_history, best_score)
                    self.move_tracker.append(best_move)
                    # update the best network
                    self.best_network = best_state
                    # if move involves a pool provider
                    # drop the new provider from pool so that the algo knows not to try that one again
                    if best_move[0] == "swap":
                        self.pool.drop(best_move[2], inplace = True)
                    elif best_move[0] == "addition":
                        self.pool.drop(best_move[1], inplace = True)
                    stop = time.perf_counter()
                    self.time_tracker = np.append(self.time_tracker, stop - start)
                    break
            if len(self.move_tracker) < self.total_optimization_rounds:
                stop = time.perf_counter()
                self.time_tracker = np.append(self.time_tracker, stop - start)
                print("No more options for optimization")
                break

            stop = time.perf_counter()
            self.time_tracker = np.append(self.time_tracker, stop - start)

        print(f"Average seconds per round of optimization: {self.time_tracker.mean().round(1)}")
        print(f"Adequacy score for best network: {self.adequacy(self.best_network)}\n")