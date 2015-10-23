# -*- coding: utf-8 -*-
from __future__ import division
from os import path, mkdir, remove
import numpy as np
from sqlite3 import connect
import warnings

############################################
#           NOTATION                       #
############################################

# For the needs of coding, we don't use systematically here the same notation as in the article.
# Here are the matches:

# For agent type:
# * '0' means a type-12 agent;
# * '1' means a type-21 agent;
# * '2' means a type-23 agent;
# * '3' means a type-32 agent;
# * '4' means a type-31 agent;
# * '5' means a type-13 agent.

# For a decision:
# * '0' means 'type-a decision';
# * '1' means 'type-b decision'.

# For a choice:
# * '0' means 'a-ij' if the agent faces a type-a decision and 'b-kj' if the agent faces a type-b decision;
# * '1' means 'a-ik'  if the agent faces a type-a decision and 'b-ki'  if the agent faces type-b decision.

# For markets,
# * '0' means the part of the market '12' where are the agents willing
#       to exchange type-1 good against type-2 good;
# * '1' means the part of the market '12' where are the agents willing
#       to exchange type-2 good against type-1 good;
# * '2' means the part of the market '23' where are the agents willing
#       to exchange type-2 good against type-3 good;
# * '3' means the part of the market '23' where are the agents willing
#       to exchange type-3 good against type-2 good;
# * '4' means the part of the market '31' where are the agents willing
#       to exchange type-3 good against type-1 good;
# * '5' means the part of the market '31' where are the agents willing
#       to exchange type-1 good against type-3 good.


class Economy(object):

    def __init__(self, workforce, alpha_value, temperature):

        self.workforce = workforce  # Number of agents by type
        self.n = np.sum(workforce)  # Total number of agents

        self.alpha = alpha_value  # Learning coefficient
        self.temperature = temperature  # Softmax parameter

        self.type = np.zeros(self.n, dtype=int)

        self.type[:] = np.concatenate(([0, ]*self.workforce[0],
                                       [1, ]*self.workforce[1],
                                       [2, ]*self.workforce[2],
                                       [3, ]*self.workforce[3],
                                       [4, ]*self.workforce[4],
                                       [5, ]*self.workforce[5]))

        # Each agent possesses an index by which he can be identified.
        #  Here are the the indexes lists corresponding to each type of agent:
        self.idx0 = np.where(self.type == 0)[0]
        self.idx1 = np.where(self.type == 1)[0]
        self.idx2 = np.where(self.type == 2)[0]
        self.idx3 = np.where(self.type == 3)[0]
        self.idx4 = np.where(self.type == 4)[0]
        self.idx5 = np.where(self.type == 5)[0]

        # The "placement array" is a 3-D matrix (d1: type, d2: decision, d3: choice).
        #  Allow us to retrieve the market where is supposed to go an agent according to:
        #  * his type,
        #  * the decision he faced,
        #  * the choice he made.
        self.placement = np.array(
            [[[0, 5],
              [3, 4]],
             [[1, 2],
              [4, 3]],
             [[2, 1],
              [5, 0]],
             [[3, 4],
              [0, 5]],
             [[4, 3],
              [1, 2]],
             [[5, 0],
              [2, 1]]])

        self.place = np.zeros(self.n, dtype=int)

        # The "decision array" is a 3D-matrix (d1: finding_a_partner, d2: decision, d3: choice).
        # Allow us to retrieve the decision faced by an agent at t according to
        #  * the fact that he succeeded in his exchange at t-1,
        #  * the decision he faced at t-1,
        #  * the choice he made at t-1.
        self.decision_array = np.array(
            [[[0, 0],
              [1, 1]],
             [[0, 1],
              [0, 0]]])

        self.decision = np.zeros(self.n, dtype=int)

        self.choice = np.zeros(self.n, dtype=int)

        self.random_number = np.zeros(self.n)  # Used for taking a decision

        self.probability_of_choosing_option0 = np.zeros(self.n, dtype=float)

        self.finding_a_partner = np.zeros(self.n, dtype=int)

        # Values for each option of choice.
        # The 'option0' and 'option1' are just the options that are reachable by the agents at time t,
        #  among the four other options.
        self.value = np.zeros(self.n, dtype=[("a-ij", float, 1),
                                             ("a-ik", float, 1),
                                             ("b-kj", float, 1),
                                             ("b-ki", float, 1),
                                             ("option0", float, 1),
                                             ("option1", float, 1)])

        # Estimation of the easiness of each type of exchange.
        self.estimation = np.zeros(self.n, dtype=[("ij", float, 1),
                                                  ("ik", float, 1),
                                                  ("kj", float, 1),
                                                  ("ki", float, 1)])

        self.estimation_array = self.estimation.view(float).reshape(self.n, 2, 2)  # Reshape of ease the computation

        # This is the initial guest (same for every agent).
        # '1' means each type of exchange can be expected to be realized in only one unit of time
        # The more the value is close to zero, the more an exchange is expected to be hard.
        self.estimation[:] = 1, 1, 1, 1

        # Preferences for each agent
        # '1' means that when facing a type-a decision, an agent attributes a higher value to a-ik (first part
        #  of an indirect exchange) than to a-ij (direct exchange).
        self.preferences = np.zeros(self.n, dtype=int)

        # Preferences by type of agent
        self.preferences_by_type = np.zeros(len(self.workforce))

    def update_decision(self):

        # Set the decision each agent faces at time t, according to the fact he succeeded or not in his exchange at t-1,
        #  the decision he previously faced, and the choice he previously made.
        self.decision[:] = self.decision_array[self.finding_a_partner,
                                               self.decision,
                                               self.choice]

    def update_options_values(self):

        # Each agent try to minimize the time to consume
        # That is v(option) = 1/(1/estimation)

        # Set value to each option choice
        self.value["a-ij"] = self.estimation["ij"]

        try:

            self.value["a-ik"] = (self.estimation["ik"] * self.estimation["kj"]) / \
                                 (self.estimation["ik"] + self.estimation["kj"])
        except RuntimeWarning:  # Avoid division by 0
            self.value["a-ik"] = 0

        self.value["b-kj"] = self.estimation["kj"]

        try:
            self.value["b-ki"] = (self.estimation["ki"] * self.estimation["ij"]) / \
                                 (self.estimation["ki"] + self.estimation["ij"])
        except RuntimeWarning:  # Avoid division by 0
            self.value["b-ki"] = 0

    def make_a_choice(self):

        id0 = np.where(self.decision == 0)[0]
        id1 = np.where(self.decision == 1)[0]

        if len(id0):  # Do matches only if there is agents that face type-a decision.

            self.value["option0"][id0] = self.value["a-ij"][id0]
            self.value["option1"][id0] = self.value["a-ik"][id0]

        if len(id1):  # Do matches only if there are agents that face type-b decision.

            self.value["option0"][id1] = self.value["b-kj"][id1]
            self.value["option1"][id1] = self.value["b-ki"][id1]

        # Set a probability to current option 0 using softmax rule
        # (As there is only 2 options each time, computing probability for a unique option is sufficient)
        self.probability_of_choosing_option0[:] = \
            np.exp(self.value["option0"]/self.temperature) / \
            (np.exp(self.value["option0"]/self.temperature) +
             np.exp(self.value["option1"]/self.temperature))

        self.random_number[:] = np.random.uniform(0., 1., self.n)  # Generate random numbers

        # Make a choice using the probability of choosing option 0 and a random number for each agent
        # Choose option 1 if random number > ou = to probability of choosing option 0,
        #  choose option 0 otherwise
        self.choice[:] = self.random_number >= self.probability_of_choosing_option0

    def who_is_where(self):

        # Place the agents according to their type, decision and choice
        self.place[:] = self.placement[self.type, self.decision, self.choice]

    def make_the_transactions(self):

        # Re-initialize the variable for succeeded exchanges
        self.finding_a_partner[:] = 0

        # Find the attendance of each part of the markets
        ipp0 = np.where(self.place == 0)[0]
        ipp1 = np.where(self.place == 1)[0]
        ipp2 = np.where(self.place == 2)[0]
        ipp3 = np.where(self.place == 3)[0]
        ipp4 = np.where(self.place == 4)[0]
        ipp5 = np.where(self.place == 5)[0]

        # Make as encounters as possible
        for ip0, ip1 in [(ipp0, ipp1), (ipp2, ipp3), (ipp4, ipp5)]:  # Consider the two parts of each market

            # If there is nobody in this particular market, do not do nothing.
            if len(ip0) == 0 or len(ip1) == 0:

                pass

            # If there is less agents in one part of the market than in the other:
            #  * agents in the less attended part get successful (that is they can proceed to an exchange);
            #  * among the agent present in the most attended part, randomly select as agents in that part of the market
            #      that there is on the other market: these selected agents can proceed to their exchange.
            elif len(ip0) < len(ip1):

                self.finding_a_partner[ip0] = 1
                np.random.shuffle(ip1)
                self.finding_a_partner[ip1[:len(ip0)]] = 1

            else:

                self.finding_a_partner[ip1] = 1
                np.random.shuffle(ip0)
                self.finding_a_partner[ip0[:len(ip1)]] = 1

    def update_estimations(self):

        # Each agent learn from the fact that he succeeds or not in his exchange,
        #   updating the estimation of the exchange that he just attempted.
        self.estimation_array[xrange(self.n), self.decision, self.choice] += \
            self.alpha * (self.finding_a_partner -
                          self.estimation_array[xrange(self.n), self.decision, self.choice])

    def update_preferences(self):

        # For each agent, we note '1' if he prefers indirect exchange, '0' otherwise
        self.preferences[:] = self.value["a-ik"] > self.value["a-ij"]

        # Compute preferences by type
        # [Ignore Warnings about np.mean calling with an empty slice
        #   (useful for economies without double coincidence of needs)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.preferences_by_type = \
                [np.mean(self.preferences[self.idx0]),
                 np.mean(self.preferences[self.idx1]),
                 np.mean(self.preferences[self.idx2]),
                 np.mean(self.preferences[self.idx3]),
                 np.mean(self.preferences[self.idx4]),
                 np.mean(self.preferences[self.idx5])]


class BackUp(object):

    def __init__(self, folder_path):

        # Backup is a database format, using Sqlite3 management system

        self.db_path = "{0}/economy.db".format(folder_path)
        self.data_table = "Preferences"
        self.connexion = None
        self.cursor = None

    def open(self):

        # Create connexion to the database
        self.connexion = connect(self.db_path)
        self.cursor = self.connexion.cursor()

    def create_table(self):

        # Create a table for saving the results
        self.open()

        query = "CREATE TABLE {0}(" \
                "TIME INT PRIMARY KEY," \
                "A12 REAL," \
                "A21 REAL," \
                "A23 REAL, " \
                "A32 REAL," \
                "A31 REAL," \
                "A13 REAL)".format(self.data_table)
        self.cursor.execute(query)
        self.close()

    def insert(self, **kwargs):

        # Insert new line of data in the database
        self.open()

        column = ""
        value = ""

        for arg in kwargs:
            column += "{0}, ".format(arg)
            value += "'{0}', ".format(kwargs[arg])

        column = column[:-2]
        value = value[:-2]

        query = "INSERT INTO {0} ({1}) VALUES ({2})".format(self.data_table, column, value)

        self.cursor.execute(query)
        self.close()

    def read(self, **kwargs):

        # Read database content.
        # 'kwargs' is expected to be a dictionary mentioning the conditions to select data to read.

        self.open()

        condition = ""

        for arg in kwargs:
            condition += "{0}={1}, ".format(arg, kwargs[arg])

        condition = condition[:-2]

        query = "SELECT `A12`, `A21`, `A23`, `A32`, `A31`, `A13` " \
                "FROM {0} WHERE {1}".format(self.data_table, condition)

        self.cursor.execute(query)
        content = self.cursor.fetchall()[0]
        self.close()

        return content

    def close(self):

        # Save modifications and close connexion.
        self.connexion.commit()
        self.connexion.close()


class MoneyDetector(object):

    def __init__(self, workforce, time_limit, observations_required, threshold, back_up):

        # Where to catch data
        self.backup = back_up

        # Economy variables required for monetary analysis
        self.workforce = workforce
        self.time_limit = time_limit

        # Criteria to use for monetary analysis
        self.observations_required = observations_required
        self.threshold = threshold

        # The different configurations for monetary states:
        #  * 'dc' configurations are for economies with Double Coincidence of needs;
        #  * 'wdc' configurations are for economies Without Double Coincidence of needs.
        # The dictionaries entries indicate the type of the monetized good ('1', '2' or '3').
        # The dictionaries values are such as:
        #  * The first element of the list indicate agent types that are supposed to prefer indirect exchanges;
        #  * The second element of the list indicate agent types that are supposed to prefer direct exchanges.
        self.configurations = \
            {"dc":
                {"1": [[2, 3], [0, 1, 4, 5]],
                 "2": [[4, 5], [0, 1, 2, 3]],
                 "3": [[0, 1], [2, 3, 4, 5]]},
             "wdc":
                {"1": [[2], [0, 4]],
                 "2": [[4], [0, 2]],
                 "3": [[0], [2, 4]]}}

    def run(self):

        # Depending on if the economy is with or without double coincidence of needs, then
        #   monetary test should not use the same configurations for testing.
        if not [self.workforce[i] == 0 for i in [1, 3, 5]].count(False):

            test_result = self.monetary_test("wdc")

        else:
            test_result = self.monetary_test("dc")

        # Return the result of the test
        return test_result

    def monetary_test(self, economy_type):

        # Variable to contain the test result.
        is_monetary = 0

        # Variable to contain the object type which emerges as money, if there is one.
        money = ""

        # Use the configurations for economies with or without double coincidence of needs,
        #   according of the type of the actual economy.
        for i in self.configurations[economy_type]:

            # We want this variable to reach the number of observations of monetary states
            #   required for considering that the economy is a monetary economy.
            conclusive_observation = 0

            # Consider economy state at a particular time for as times units as
            #   observations of monetary states are required for considering that the economy is a monetary economy.
            # (By default, the last 100 economy states have to be monetary states to consider that the economy
            #   is a monetary economy).
            for j in range(1, self.observations_required + 1):

                # Began to consider the preferences at the last time unit, and the penultimate, and so on.
                preferences = self.backup.read(TIME=self.time_limit-j)

                # If the economy state corresponds to a monetary state...
                if not [preferences[k] >= (1 - self.threshold)
                        for k in self.configurations[economy_type][i][0]]. count(False) \
                    and not [preferences[l] <= self.threshold
                             for l in self.configurations[economy_type][i][1]].count(False):

                    # ...increment the counter of observations of a monetary state
                    conclusive_observation += 1

                # If not, it is useless to continue the test for this particular good,
                #   so pass to the next configuration to test if an another good has emerged as money
                else:
                    break

            # If there is as monetary states that have been observed than observations of monetary states required
            #   for considering that the economy is a monetary economy...
            if conclusive_observation == self.observations_required:

                # ...then this economy is indeed a monetary economy.
                is_monetary = 1

                # Good corresponding to this particular configuration permits to have as monetary states observation
                #  as required. It is then considered as the money of this economy.
                money = i

                # Since we already found the money, it is useless to test for an another good,
                #   so we can break the loop and return the result.
                break

        # Return the result of the test
        return {"is monetary": is_monetary, "money": money}


class SimulationRunner(object):

    def __init__(self, workforce, alpha, temperature, time_limit, back_up):

        # Backup support
        self.backup = back_up

        # Time the simulation should last
        self.time_limit = time_limit

        # Create the economy to simulate
        self.eco = Economy(workforce=workforce, alpha_value=alpha, temperature=temperature)

    def run(self):

        # Create a table in a database to contain data
        self.backup.create_table()

        # Set time counter at 0.
        t = 0

        # Run simulation for as time units as required.
        while t < self.time_limit:

            # Make agents updating the decision they are facing
            self.eco.update_decision()

            # Make agents updating the values they attribute to options
            self.eco.update_options_values()

            # Make agents choosing
            self.eco.make_a_choice()

            # Move the agents where they are supposed to go
            self.eco.who_is_where()

            # Realize the transactions in the different markets
            self.eco.make_the_transactions()

            # Make agents learn about the success rates of each type of exchange
            self.eco.update_estimations()

            # Update the agents preferences in order to save them
            self.eco.update_preferences()

            # Make backup of the preferences
            self.backup.insert(TIME=t,
                               A12=self.eco.preferences_by_type[0],
                               A21=self.eco.preferences_by_type[1],
                               A23=self.eco.preferences_by_type[2],
                               A32=self.eco.preferences_by_type[3],
                               A31=self.eco.preferences_by_type[4],
                               A13=self.eco.preferences_by_type[5])

            # Increment the time counter
            t += 1


if __name__ == "__main__":

    ##################################
    #        Global variables        #
    ##################################

    # Set the fundamental structure of the economy. 
    #  Here is an example of fundamental structure inducing the emergence of the good '2' as money 
    #  (others things being equal).
    #  For instance, a fundamental structure such as [100, 0, 100, 0, 100, 0] will produce a negative result,
    #  since no money will emerge.
    eco_workforce = [100, 0, 100, 0, 300, 0]
    # Set the coefficient learning.
    eco_alpha = 0.5
    # Set the softmax parameter.
    eco_temperature = 0.05
    # Set the number of time units the simulation will run.
    eco_time_limit = 10000
    # Set the number of time units used to check if the economy is in a monetary state.
    analysis_observation_required = 100
    # Set the tolerance threshold concerning the proportion of agents of a same type
    #   "authorized" to deviate from what would predict a pure Nash equilibrium
    analysis_threshold = .10
    # Folder in which the results will be inserted.
    folder = 'Simulation/'

    ##################################
    #   Beginning of the program     #
    ##################################

    # If the above-mentioned folder doesn't exist, create it;
    #  otherwise, remove the database containing previous data.
    if not path.exists(folder):
        mkdir(folder)
    else:
        remove("Simulation/economy.db")

    # Create a backup support
    backup = BackUp(folder)

    # Create a "simulation runner" that will manage simulation.
    simulation_runner = SimulationRunner(workforce=eco_workforce, alpha=eco_alpha, temperature=eco_temperature,
                                         time_limit=eco_time_limit, back_up=backup)

    # Ask the "simulation runner" to launch the simulation.
    print "Producing data (it can take time)..."
    simulation_runner.run()

    # Create a "money detector" that will analyse if the simulated economy became a monetary one.
    money_detector = MoneyDetector(workforce=eco_workforce, time_limit=eco_time_limit,
                                   observations_required=analysis_observation_required,
                                   threshold=analysis_threshold, back_up=backup)

    # Ask the chief operating officer to analyse if the simulated economy can be considered as a monetary one.
    # Then print the result.
    print "Test the simulated economy for money emergence..."

    result = money_detector.run()

    if result["is monetary"] == 1:
        print "The simulated economy appears to be a monetary economy " \
              "with the good {0} as money.".format(result["money"])
    else:
        print "The simulated economy appears to not be a monetary economy."

    ############################
    #   End of the program     #
    ############################
