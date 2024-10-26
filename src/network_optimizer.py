import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from sklearn.metrics.pairwise import haversine_distances
import random
import time
import math

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

        # create access listing
        if len(self.initial_network) > 0:
            self.access_listing = self.members.rename(
                {"latitude":"member_latitude", 
                "longitude":"member_longitude",
                "county":"member_county"}, axis=1).merge(pd.concat([self.initial_network, self.initial_pool]).rename(
                    {"latitude":"provider_latitude",
                    "longitude":"provider_longitude",
                    "county":"provider_county"}, axis=1), how="cross")

        else:
            self.access_listing = self.members.rename(
                {"latitude":"member_latitude", 
                "longitude":"member_longitude",
                "county":"member_county"}, axis=1).merge(self.initial_pool.rename(
                    {"latitude":"provider_latitude",
                    "longitude":"provider_longitude",
                    "county":"provider_county"}, axis=1), how="cross")

        def haversine(lat1, lon1, lat2, lon2):
     
            # distance between latitudes
            # and longitudes
            dLat = (lat2 - lat1) * math.pi / 180.0
            dLon = (lon2 - lon1) * math.pi / 180.0
        
            # convert to radians
            lat1 = (lat1) * math.pi / 180.0
            lat2 = (lat2) * math.pi / 180.0
        
            # apply formulae
            a = (pow(math.sin(dLat / 2), 2) +
                pow(math.sin(dLon / 2), 2) *
                    math.cos(lat1) * math.cos(lat2))
            rad = 3959
            c = 2 * math.asin(math.sqrt(a))
            return rad * c
        
        self.access_listing["distance"] = self.access_listing.apply(lambda row: haversine(row.member_latitude, 
                                                                                          row.member_longitude, 
                                                                                          row.provider_latitude, 
                                                                                          row.provider_longitude), axis=1).round(1)

        self.access_listing = self.access_listing.merge(self.adequacy_reqs, left_on=["member_county", "specialty"], right_on=["county", "specialty"], how="left")
        
        self.pct_serving = (self.access_listing[["location_id"]].merge(
            (self.access_listing.loc[self.access_listing.distance <= self.access_listing.distance_req].groupby("location_id")["member_id"].count() / len(self.members))
            .reset_index()
            .rename({"member_id":"pct_serving"}, axis=1), on="location_id", how="left").fillna(0).drop_duplicates().reset_index(drop=True))

        self.pool = pool.copy().merge(self.pct_serving, on=["location_id"], how="left")

        self.best_network = network.copy().merge(self.pct_serving, on=["location_id"], how="left") if network is not None else pd.DataFrame(columns=self.pool.columns)

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
            network_distances = (self.access_listing[["member_id", "npi", "location_id", "distance", "provider_count"]]
                     .merge(network[["npi", "location_id", "specialty", "county"]], on=["npi", "location_id"], how="inner")
                     .merge(self.adequacy_reqs[["county", "specialty", "distance_req"]], on=["county", "specialty"], how="left"))

            network_distances["meets_distance"] = network_distances.distance <= network_distances.distance_req

            access_detail = (network_distances
                         .loc[network_distances.meets_distance == True]
                         .groupby(["member_id", "county", "specialty", "provider_count"])[["npi"]]
                         .count()
                         .reset_index()
                         .rename({"npi":"servicing_provider_count"}, axis=1))
            
            member_access_summary = (access_detail
                         .loc[access_detail.servicing_provider_count >= access_detail.provider_count]
                         .groupby(["county", "specialty"])[["member_id"]]
                         .nunique()
                         .reset_index()
                         .rename({"member_id":"members_with_access"}, axis=1))
            
            servicing_provider_summary = (network_distances
                              .groupby(["county", "specialty"])[["npi"]]
                              .nunique()
                              .reset_index()
                              .rename({"npi":"servicing_providers"}, axis=1))
            
            adequacy_detail = (self.adequacy_reqs.merge(self.members
                                .groupby("county")[["member_id"]]
                                .count()
                                .reset_index()
                                .rename({"member_id":"total_members"}, axis=1), how="left", on="county")
                               .merge(member_access_summary, how="left", on=["county", "specialty"])
                               .merge(servicing_provider_summary, how="left", on=["county", "specialty"])
                               .fillna(0))
            
            adequacy_detail["pct_with_access"] = adequacy_detail.members_with_access / adequacy_detail.total_members

            adequacy_detail["adequacy_index"] = adequacy_detail.pct_with_access * (adequacy_detail.servicing_providers / adequacy_detail.min_providers).apply(lambda x: min(1, x))

            self.adequacy_detail = adequacy_detail
            
            return adequacy_detail.adequacy_index.mean()
        
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
        additions = [("addition", group) for group in pool.group_id.drop_duplicates().to_list()]
        return additions
    
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
            new_network = pd.concat([new_network, pool.loc[pool.group_id == change[1]]]).reset_index(drop=True)
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

            new_states = [self.create_state(self.best_network, self.pool, change) for change in state_changes]
            new_scores = [self.objective(state) for state in new_states]
            best_score = np.max(new_scores)
            best_state_idx = np.argmax(new_scores)
            best_move = state_changes[best_state_idx]
            best_state = new_states[best_state_idx]
            
            if best_score > self.performance_history[-1]:
                self.performance_history = np.append(self.performance_history, best_score)
                self.move_tracker.append(best_move)
                self.best_network = best_state
                if best_move[0] == "addition":
                        change_members = self.pool.loc[self.pool.group_id == best_move[1]].index
                        self.pool.drop(change_members, inplace = True)
                
            else:
                stop = time.perf_counter()
                self.time_tracker = np.append(self.time_tracker, stop - start)
                print("No more options for optimization")
                break

            stop = time.perf_counter()
            self.time_tracker = np.append(self.time_tracker, stop - start)

        print(f"Average seconds per round of optimization: {self.time_tracker.mean().round(1)}")
        print(f"Adequacy Index for best network: {round(self.adequacy(self.best_network), 2)}\n")