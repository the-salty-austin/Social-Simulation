import datetime
import random
import numpy as np
from scipy import integrate, stats

import csv

# parameters
MEMORIES = [0.0,0.2,0.4,0.6,0.8]
COSTS = [0.0,0.2,0.4,0.6,0.8]
ROUNDS = 25

# generate Gaussian function
def gauss(x, mu=0, sigma=1):
    return stats.norm.pdf(x, mu, sigma)

class Grid():
    def __init__(self, size=32) -> None:
        tmp = [None]*size
        for i in range(size):
            tmp[i] = [False]*size
        self.grid = tmp

    def assign_voter_location(self, x, y, individual):
        self.grid[y][x] = individual


class Voter():
    def __init__(self) -> None:
        '''
        preference: pre-determined \n
        prev_utilities: [ U received this time ]\n
        voted: [ whether one votes this time ]\n
        vote_choices: [which party one chose. 0=no vote / 1=left / 2=right]\n
        neighbor_records: [ 0: None, 1~ record ]
        '''
        self.preference = np.random.standard_normal()
        self.prev_utilities = []
        self.voted = [True if random.randint(0,100) < 51 else False]
        self.vote_choices = []
        self.neighbor_records = [None]

class Party():
    def __init__(self, platform) -> None:
        '''
        platforms \n
        votes: [how many votes received in each election]\n
        wins: [whether the party has won (True) or not(False).]
        '''
        self.platforms = [platform]  # between -1 and 1
        self.votes = []
        self.wins = []


for COST in COSTS:
    for MEMORY in MEMORIES:
        # FIRST ELECTION (initialization)
        g = Grid()

        # initialize all 1024 individuals
        for y in range(32):
            for x in range(32):
                individual = Voter()
                g.assign_voter_location( x, y, individual)

        # initialize 2 parties
        party_neg = Party(-1)
        party_pos = Party(1)


        start_time = datetime.datetime.now()
        # >>>>>>>>>>> FIRST ELECTION <<<<<<<<<<<<<<<<<<<
        vote_neg, vote_pos = 0, 0

        for y in range(32):
            for x in range(32):
                individual = g.grid[y][x]

                if individual.voted[0] is True:
                    # vote neg (choice 1)
                    u_neg = -(individual.preference - party_neg.platforms[0])**2 - COST
                    # vote pos (choice 2)
                    u_pos = -(individual.preference - party_pos.platforms[0])**2 - COST

                    if u_neg >= u_pos:
                        vote_neg += 1
                        individual.prev_utilities.append(u_neg)
                        individual.vote_choices.append(1)
                    else:
                        vote_pos += 1
                        individual.prev_utilities.append(u_pos)
                        individual.vote_choices.append(2)

                else:
                    u_neg = -(individual.preference - party_neg.platforms[0])**2
                    u_pos = -(individual.preference - party_pos.platforms[0])**2
                    if u_neg >= u_pos: u = u_neg
                    else: u = u_pos
                    individual.prev_utilities.append(u)
                    individual.vote_choices.append(0)

        if vote_neg > vote_pos:
            party_neg.wins.append( True )  # elected
            party_pos.wins.append( False )
        elif vote_neg < vote_pos:
            party_pos.wins.append( True )  # elected
            party_neg.wins.append( False )
        else:
            party_pos.wins.append( False )
            party_neg.wins.append( False )

        party_neg.votes.append( vote_neg )
        party_pos.votes.append( vote_pos )

        # write to log file
        # PARTY >>> [#election, #party, COST, platform, elected?]
        with open(f'./log/party_log_C{int(COST*10)}M{int(MEMORY*10)}.csv', 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow( [1, 1, COST, MEMORY, party_neg.platforms[0], party_neg.wins[0] ] )
            writer.writerow( [1, 2, COST, MEMORY, party_pos.platforms[0], party_pos.wins[0] ] )

        # VOTER >>> [#election, #voter, COST, neightbor cnt, voter cnt, abstainer cnt]
        with open(f'./log/voters_log_C{int(COST*10)}M{int(MEMORY*10)}.csv', 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            idx = 0
            for y in range(32):
                for x in range(32):
                    idx += 1
                    individual = g.grid[y][x]
                    c = individual.vote_choices[0]
                    writer.writerow( [1, idx, COST, MEMORY, individual.voted[0], -1, -1, c] )

        print(f'[Election 1 Ended] Time Elapsed: {datetime.datetime.now()-start_time}')

        # >>>>>>>>>>>> SECOND+ ELECTION <<<<<<<<<<<<<<<<<<<

        for r in range(1, ROUNDS):
            
            # both parties try to adapt to previous results.
            neg_vote_cnt = party_neg.votes[r-1]
            pos_vote_cnt = party_pos.votes[r-1]
            try: s = neg_vote_cnt / (neg_vote_cnt+pos_vote_cnt)
            except ZeroDivisionError: s = 0
            x1 = party_neg.platforms[r-1]
            x2 = party_pos.platforms[r-1]
            min_err = 100
            best_m = 0
            lowerbound = -np.inf
            upperbound = (x1+x2)/2.0
            
            m_guess = -2
            for i in range(201):
                m_guess = m_guess + 0.01*i
                if m_guess < -1 or m_guess > 1:
                    m_guess += 0.1

                # integrate between bounds
                integral = integrate.quad(gauss, lowerbound, upperbound, args=(m_guess) )[0]
                err = np.abs(integral-s)
                if err < min_err:
                    min_err = err
                    best_m = m_guess

            print('s >>>> ', s, best_m)
            party_neg.platforms.append( best_m*(1-s) + party_neg.platforms[r-1]*(s) )
            party_pos.platforms.append( best_m*(s) + party_pos.platforms[r-1]*(1-s) )
            # party optimization ended

            # individuals seek opinions from neighbors and then vote
            vote_neg, vote_pos = 0, 0

            # first, find all neighbors
            for y in range(32):
                for x in range(32):
                    individual = g.grid[y][x]
                    # corner case
                    if (x==0 and y==0):
                        all_neighbors = [ g.grid[0][1] , g.grid[1][0] , g.grid[1][1] ]
                    elif (x==31 and y==0):
                        all_neighbors = [ g.grid[0][30] , g.grid[1][31] , g.grid[1][30] ]
                    elif (x==0 and y==31):
                        all_neighbors = [ g.grid[30][0] , g.grid[31][1] , g.grid[30][1] ]
                    elif (x==31 and y==31):
                        all_neighbors = [ g.grid[31][30] , g.grid[30][31] , g.grid[30][30] ]
                    # edge (not corner) case
                    elif (y==0 and 1<=x<=30):
                        all_neighbors = [ g.grid[0][x-1] , g.grid[0][x+1] ] + g.grid[1][x-1:x+2]
                    elif (y==31 and 1<=x<=30):
                        all_neighbors = [ g.grid[31][x-1] , g.grid[31][x+1] ] + g.grid[30][x-1:x+2]
                    elif (1<=y<=30 and x==0):
                        all_neighbors = [ g.grid[y-1][0] , g.grid[y+1][0] ]
                        for array in g.grid[y-1:y+2]:
                            all_neighbors.append( array[1] )
                    elif (1<=y<=30 and x==31):
                        all_neighbors = [ g.grid[y-1][31] , g.grid[y+1][31] ]
                        for array in g.grid[y-1:y+2]:
                            all_neighbors.append( array[30] )
                    # inside case:
                    else:
                        n_above = g.grid[y-1][x-1:x+2]
                        n_middle = g.grid[y][x-1:x+2]
                        n_below = g.grid[y+1][x-1:x+2]
                        all_neighbors = n_above + n_middle + n_below

                    voter_tot_u = 0
                    voter_cnt = 0
                    abstainer_tot_u = 0
                    abstainer_cnt = 0
                    for neighbor in all_neighbors:
                        if neighbor.voted[r-1] is True:
                            # neighbor voted in the prev election
                            voter_tot_u += neighbor.prev_utilities[r-1]
                            voter_cnt += 1
                        elif neighbor.voted[r-1] is False:
                            # neighbor abstained in the prev election
                            abstainer_tot_u += neighbor.prev_utilities[r-1]
                            abstainer_cnt += 1
                    
                    if voter_cnt == 0 or abstainer_cnt == 0:
                        # if there is either no voters or abstainers within one's nbhd
                        # one remains the previous decision
                        individual.voted.append( individual.voted[r-1] )
                        individual.vote_choices.append( individual.vote_choices[r-1] )
                        if individual.voted[r-1] is True:
                            u = -(individual.preference - individual.vote_choices[r-1])**2 - COST
                            individual.prev_utilities.append(u)
                        else:
                            u = -(individual.preference - individual.vote_choices[r-1])**2
                            individual.prev_utilities.append(u)
                        individual.neighbor_records.append( (voter_cnt, abstainer_cnt) )
                        continue
                    
                    try:
                        vote_avg = voter_tot_u / voter_cnt
                    except ZeroDivisionError:
                        vote_avg = 0
                    try:
                        abstain_avg = abstainer_tot_u / abstainer_cnt
                    except ZeroDivisionError:
                        abstain_avg = 0

                    # compare whether voting (which party) or abstaining
                    # yields best utility.
                    
                    # if vote
                    # >>> vote neg (choice 1)
                    u_neg = -(individual.preference - party_neg.platforms[r])**2 - COST
                    u_vote_neg = (MEMORY)*vote_avg + (1-MEMORY)*u_neg
                    # >>> vote pos (choice 2)
                    u_pos = -(individual.preference - party_pos.platforms[r])**2 - COST
                    u_vote_pos = (MEMORY)*vote_avg + (1-MEMORY)*u_pos

                    if u_vote_neg > u_vote_pos:
                        vote_max_u = u_vote_neg
                    else:
                        vote_max_u = u_vote_pos
                    
                    # if abstain
                    # clearly, dist to the nearest party platform
                    u_neg = -(individual.preference - party_neg.platforms[r])**2
                    u_abstain_neg = (1-MEMORY)*abstain_avg + MEMORY*u_neg
                    u_pos = -(individual.preference - party_pos.platforms[r])**2
                    u_abstain_pos = (1-MEMORY)*abstain_avg + MEMORY*u_pos
                    #
                    if u_abstain_neg > u_abstain_pos:
                        abstain_max_u = u_abstain_neg
                    else:
                        abstain_max_u = u_abstain_pos
                    #
                    if vote_max_u > abstain_max_u:
                        want_to_vote = True
                    else:
                        want_to_vote = False

                    individual.voted.append( want_to_vote )

                    # 
                    if want_to_vote and (u_vote_neg <= u_vote_pos):
                        individual.prev_utilities.append(u_vote_pos)
                        individual.vote_choices.append(2)
                        vote_pos += 1
                            
                    if want_to_vote and (u_vote_neg > u_vote_pos):
                        individual.prev_utilities.append(u_vote_neg)
                        individual.vote_choices.append(1)
                        vote_neg += 1

                    elif not want_to_vote and (u_abstain_neg >= u_abstain_pos):
                        individual.prev_utilities.append(u_abstain_neg)
                        individual.vote_choices.append(0)
                    elif not want_to_vote and (u_abstain_neg < u_abstain_pos):
                        individual.prev_utilities.append(u_abstain_pos)
                        individual.vote_choices.append(0)

                    individual.neighbor_records.append( (voter_cnt, abstainer_cnt) )

            if vote_neg > vote_pos:
                party_neg.wins.append( True )
                party_pos.wins.append( False )
            elif vote_neg < vote_pos:
                party_pos.wins.append( True )
                party_neg.wins.append( False )
            else:
                party_pos.wins.append( False )
                party_neg.wins.append( False )

            party_neg.votes.append( vote_neg )
            party_pos.votes.append( vote_pos )

            # write to log file
            # PARTY >>> [#election, #party, COST, platform, elected?]
            with open(f'./log/party_log_C{int(COST*10)}M{int(MEMORY*10)}.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow( [r+1, 1, COST, MEMORY, party_neg.platforms[r], party_neg.wins[r] ] )
                writer.writerow( [r+1, 2, COST, MEMORY, party_pos.platforms[r], party_pos.wins[r] ] )

            # VOTER >>> [#election, #voter, COST, voted?, voter cnt, abstainer cnt]
            with open(f'./log/voters_log_C{int(COST*10)}M{int(MEMORY*10)}.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                idx = 0
                for y in range(32):
                    for x in range(32):
                        idx += 1
                        individual = g.grid[y][x]
                        v_decision = individual.voted[r]
                        v, a = individual.neighbor_records[r]
                        c = individual.vote_choices[r]
                        
                        writer.writerow( [r+1, idx, COST, MEMORY, v_decision, v, a, c] )

            print(f'[Election {r+1} Ended] Time Elapsed: {datetime.datetime.now()-start_time}')
