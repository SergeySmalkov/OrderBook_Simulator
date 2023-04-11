#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 13:38:08 2022

@author: cheytakker
"""

import pandas as pd
import numpy as np
import math
from itertools import compress

class Simulator:
    class OB:
        def __init__(self, current_time, bids, asks):
            self.current_time = current_time
            self.bids = pd.DataFrame(bids, columns = ["P", "Q", "i"])
            self.asks = pd.DataFrame(asks, columns = ["P", "Q", "i"])
            self.best_bid = {"bb_index" : len(self.bids) - 1, "bb_price" : 0.0, 
                             "bb_i" : self.bids["i"][len(self.bids) - 1]}
            self.best_ask = {"ba_index" : len(self.asks) - 1, "ba_price" : 0.0, 
                             "ba_i" : self.asks["i"][len(self.asks) - 1]}
            self.first_lvls = {"bid_i" : 0, "ask_i" : 0, "bid_index" : 0, "ask_index": 0} # i - relatively to mid, index - in dataframe
            self.mid = 0.0
        
            self.refresh_lvls()
            
            
        def process_market_order(self, market_order):
            market_order = market_order.to_dict()
            market_order["Unexecuted_Q"] = market_order.pop("Q")
            market_order["Executed_Q"] = 0.0
            market_order["VWAP"] = 0.0 
            
            if (market_order["d"] == 1.0):
                lvl = self.best_ask["ba_index"]
                while (market_order["Unexecuted_Q"] != 0.0):
                    if (len(self.asks) == lvl): break
                    if (self.asks.loc[lvl, "Q"] == 0.0): 
                        lvl += 1 
                        continue
                    if (self.asks.loc[lvl, "Q"] >= market_order["Unexecuted_Q"]):

                        p1 = market_order["VWAP"]
                        q1 = market_order["Executed_Q"]
                        p2 = self.asks.loc[lvl, "P"]
                        q2 = market_order["Unexecuted_Q"]
                        self.asks.loc[lvl, "Q"] -= q2
                        market_order["Executed_Q"] += q2                                   
                        market_order["VWAP"] = (p1 * q1 + p2 * q2) / (q1 + q2)       
                        market_order["Unexecuted_Q"] = 0.0 
                                 
                    elif (self.asks.Q[lvl] < market_order["Unexecuted_Q"]):

                        p1 = market_order["VWAP"]
                        q1 = market_order["Executed_Q"]
                        p2 = self.asks.loc[lvl, "P"]
                        q2 = self.asks.loc[lvl, "Q"]
                        self.asks.loc[lvl, "Q"] = 0.0
                        market_order["Executed_Q"] += q2                                   
                        market_order["VWAP"] = (p1 * q1 + p2 * q2) / (q1 + q2)       
                        market_order["Unexecuted_Q"] -= q2                                   
                        lvl += 1
                        
            elif (market_order["d"] == -1.0):
                lvl = self.best_bid["bb_index"]
                while (market_order["Unexecuted_Q"] != 0.0  ):
                    if (len(self.bids) == lvl): break
                    if (self.bids.Q[lvl] == 0.0):
                        lvl += 1
                        continue
                    if (self.bids.loc[lvl, "Q"] >= market_order["Unexecuted_Q"]):

                        p1 = market_order["VWAP"]
                        q1 = market_order["Executed_Q"]
                        p2 = self.bids.loc[lvl, "P"]
                        q2 = market_order["Unexecuted_Q"]
                        self.bids.loc[lvl, "Q"]-= q2
                        market_order["Executed_Q"] += q2                                   
                        market_order["VWAP"] = (p1 * q1 + p2 * q2) / (q1 + q2)      
                        market_order["Unexecuted_Q"] = 0.0                                   
                    elif (self.bids.loc[lvl, "Q"] < market_order["Unexecuted_Q"]):

                        p1 = market_order["VWAP"]
                        q1 = market_order["Executed_Q"]
                        p2 = self.bids.loc[lvl, "P"]
                        q2 = self.bids.loc[lvl, "Q"]
                        self.bids.loc[lvl, "Q"] = 0.0
                        market_order["Executed_Q"] += q2                                   
                        market_order["VWAP"] = (p1 * q1 + p2 * q2) / (q1 + q2)       
                        market_order["Unexecuted_Q"] -= q2                                   
                        lvl += 1
            
            self.refresh_lvls()
            return market_order
        
        def process_cancelation(self, cncl_order):
            if (cncl_order["d"] == -1):
                index = self.first_lvls["ask_index"] + cncl_order.i
                if(index < len(self.asks)) :
                    self.asks.loc[index, "Q"] = max(0.0, self.asks.loc[index, "Q"] - cncl_order.Q)
                    if(cncl_order.i == 0): self.refresh_lvls()     #if ToB was changed - refresh the indexes of OB
                else: return
                    
                    
            elif(cncl_order["d"] == 1):
                
                index = self.first_lvls["bid_index"] + cncl_order.i
                if(index < len(self.bids)) :
                    self.bids.loc[index, "Q"] = max(0.0, self.bids.loc[index, "Q"] - cncl_order.Q)
                    if(cncl_order.i == 0): self.refresh_lvls()     #if ToB was changed - refresh the indexes of OB
                else: return
                
        def refresh_lvls(self):
            if(len(self.bids[self.bids["Q"] != 0.0]) == 0): print("Attention - Empty Order Book")
            if(len(self.asks[self.asks["Q"] != 0.0]) == 0): print("Attention - Empty Order Book")
            self.best_bid["bb_index"] = self.bids[self.bids["Q"] != 0.0]["P"].idxmax()
            self.best_ask["ba_index"] = self.asks[self.asks["Q"] != 0.0]["P"].idxmin()
            self.best_bid["bb_i"] = self.bids["i"][self.best_bid["bb_index"]]
            self.best_ask["ba_i"] = self.asks["i"][self.best_ask["ba_index"]]
            self.best_bid["bb_price"] = self.bids["P"][self.best_bid["bb_index"]]
            self.best_ask["ba_price"] = self.asks["P"][self.best_ask["ba_index"]]
            self.mid = round((self.best_bid["bb_price"] + self.best_ask["ba_price"]) / 2, 5)

            mid_ind = (self.best_ask["ba_i"] - self.best_bid["bb_i"]) / 2
            self.first_lvls["ask_i"] = int(np.ceil(mid_ind))
            self.first_lvls["bid_i"] = int(-np.floor(mid_ind))
            self.first_lvls["ask_index"] = int(self.asks.index[self.asks['i'] == self.first_lvls["ask_i"]][0])
            self.first_lvls["bid_index"] = int(self.bids.index[self.bids['i'] == self.first_lvls["bid_i"]][0])
            
                
        def process_limit_order(self, limit_order):
            
            if (limit_order["d"] == -1):
                index = self.first_lvls["ask_index"] + limit_order.i

                if(index < len(self.asks)) :
                    self.asks.loc[index, "Q"] = self.asks.loc[index, "Q"] + limit_order.Q
                    if(index < self.best_ask["ba_index"]): self.refresh_lvls()     #if ToB was changed - refresh the indexes of OB
                else: return
                
            elif (limit_order["d"] == 1):
                index = self.first_lvls["bid_index"] + limit_order.i

                if(index < len(self.bids)) :
                    self.bids.loc[index, "Q"] = self.bids.loc[index, "Q"] + limit_order.Q
                    if(index < self.best_bid["bb_index"]): self.refresh_lvls()     #if ToB was changed - refresh the indexes of OB
                else: return
                
                
                
            
    def __init__(self, time, lambda_0_market, a, var_e, q_0, var_q, b, start_mid, depth, step, density, lambda_0_limit, lambda_0_cncl):
        self.time = time         # time horizon for which the trading occurs
        self.lambda_0_market = lambda_0_market # starting lambda - average number of orders per second
        self.a = a               # mean reversion coefficient for lambda
        self.var_e = var_e       # variance of error for lambda
        self.q_0 = q_0           # average deal quantity
        self.var_q = var_q       # variance of corresponding normal for deal quantity 
        self.b = b               # coefficient for direction correlation
        self.shelf = 3
        
        self.start_mid = start_mid # start mid
        self.depth = depth         # depth of the order book
        self.step = step           # step size of the price
        self.density = density     # density of the order book
        
        self.lambda_0_limit = lambda_0_limit
        self.lambda_0_cncl = lambda_0_cncl
        
        self.OB_0 = self.start_OB()
        
        self.market_orders_report = pd.DataFrame(columns = ["t", "d", "Unexecuted_Q", "Executed_Q", "VWAP"])
        self.market_orders = pd.DataFrame(self.simulate_market_orders(), columns = ["t", "d", "Q"])
        self.limit_orders = pd.DataFrame(self.simulate_limit_orders(self.lambda_0_limit), columns = ["t", "d", "Q", "i"]) 
        self.cancelations = pd.DataFrame(self.simulate_limit_orders(self.lambda_0_cncl), columns = ["t", "d", "Q", "i"])
        
        self.order_books = self.matching()
        
    def start_OB(self):
        lvls_bids = np.random.choice(range(1, self.depth + 1), round(self.depth * self.density) , replace=False)
        lvls_bids.sort()
        lvls_asks = np.random.choice(range(1, self.depth + 1), round(self.depth * self.density) , replace=False)
        lvls_asks.sort()

        bids = [[self.start_mid - i  * self.step, 
                      round(np.random.lognormal(np.log(self.q_0) - self.var_q / 2, math.sqrt(self.var_q)), 4) 
                      if i in lvls_bids else 0.0, i] 
        for i in range(-self.depth * self.shelf , self.depth * self.shelf + 1)]
    
        asks = [[self.start_mid + i  * self.step, 
                      round(np.random.lognormal(np.log(self.q_0) - self.var_q / 2, math.sqrt(self.var_q)),4) 
                      if i in lvls_asks else 0.0, i]
                      for i in range(-self.depth * self.shelf , self.depth * self.shelf + 1)]
    
    
        ob = self.OB(0, bids, asks)
        return ob
     
    def simulate_market_orders(self):
        lambdas_market = []
        lambdas_market.append(self.lambda_0_market)
        times = []
        times.append(0)
        x_list = []
        quantities = []
        x_list.append(0)
        market_orders = []
         
        while (True):
            new_time = np.random.exponential(1 / lambdas_market[-1])
            if (new_time + times[-1] > self.time): break
                # recall that mean of lognormal is exp(mu + sigma^2/2)
            quantities.append(round(np.random.lognormal(np.log(self.q_0) - self.var_q / 2, math.sqrt(self.var_q)), 4)) 
            times.append(new_time + times[-1])
            x_list.append(self.b * x_list[-1] + np.random.normal(0, 1))
            lambdas_market.append(abs(lambdas_market[-1] + self.a * (self.lambda_0_market - lambdas_market[-1]) + 
                                    np.random.normal(0, math.sqrt(self.var_e))))
        times.pop(0)
        x_list.pop(0)
        directions = np.sign(x_list)
        market_orders = list(zip(times, directions, quantities))
        return market_orders
    
    def simulate_limit_orders(self, lambda_):
        bid_lambdas_limit = []
        bid_lambdas_limit.append(lambda_)
        bid_times = []
        bid_times.append(0)
        bid_lvl_num_list = []
        bid_quantities = []
        
        ask_lambdas_limit = []
        ask_lambdas_limit.append(lambda_)
        ask_times = []
        ask_times.append(0)
        ask_lvl_num_list = []
        ask_quantities = []
        
        limit_orders = []
        
        while (True):
            new_time = np.random.exponential(1 / bid_lambdas_limit[-1])
            if (new_time + bid_times[-1] > self.time): break

            bid_quantities.append(round(np.random.lognormal(np.log(self.q_0) - self.var_q / 2, math.sqrt(self.var_q)),4)) 
            bid_times.append(new_time + bid_times[-1])
            bid_lvl_num_list.append(np.random.choice(self.depth, replace=False))
            
            bid_lambdas_limit.append(abs(bid_lambdas_limit[-1] + self.a * (lambda_ - bid_lambdas_limit[-1]) + 
                                    np.random.normal(0, math.sqrt(self.var_e))))
            
        while (True):
            new_time = np.random.exponential(1 / ask_lambdas_limit[-1])
            if (new_time + ask_times[-1] > self.time): break

            ask_quantities.append(round(np.random.lognormal(np.log(self.q_0) - self.var_q / 2, math.sqrt(self.var_q)),4)) 
            ask_times.append(new_time + ask_times[-1])
            ask_lvl_num_list.append(np.random.choice(self.depth, replace=False))
            
            ask_lambdas_limit.append(abs(ask_lambdas_limit[-1] + self.a * (lambda_ - ask_lambdas_limit[-1]) + 
                                    np.random.normal(0, math.sqrt(self.var_e))))
        
        bid_times.pop(0)
        ask_times.pop(0)
        limit_orders = list(zip(bid_times, np.full(len(bid_times), 1), bid_quantities, bid_lvl_num_list))
        limit_orders = limit_orders + list(zip(ask_times, np.full(len(ask_times), -1), ask_quantities, ask_lvl_num_list))
        limit_orders.sort()
        return limit_orders

    def matching(self):
        market_index = 0
        cncl_index = 0
        limit_index = 0
        len_m = len(self.market_orders)
        len_c = len(self.cancelations)
        len_l = len(self.limit_orders)
        self.market_orders.loc[len(self.market_orders)] = [self.time + 1, -1.0, 1]    #breakpoint row,  will be deleted
        self.cancelations.loc[len(self.cancelations)] = [self.time + 1, -1.0, 1, 0]   #breakpoint row,  will be deleted
        self.limit_orders.loc[len(self.limit_orders)] = [self.time + 1, -1.0, 1, 0]   #breakpoint row,  will be deleted
       
        
        order_books = []
        order_books.append((0, self.OB_0))
        
        while ((market_index != len_m) & (cncl_index != len_c) & (limit_index != len_l)):
            if ((self.market_orders.t[market_index] <= self.cancelations.t[cncl_index]) 
            & (self.market_orders.t[market_index] <= self.limit_orders.t[limit_index] )):
                
                cur_ob = self.OB(self.market_orders.t[market_index], order_books[-1][1].bids, order_books[-1][1].asks)
                self.market_orders_report.loc[len(self.market_orders_report)] = cur_ob.process_market_order(self.market_orders.loc[market_index])
                order_books.append((self.market_orders.t[market_index], cur_ob))
                market_index += 1
                
            elif ((self.cancelations.t[cncl_index] <=  self.market_orders.t[market_index])
            & (self.cancelations.t[cncl_index] <= self.limit_orders.t[limit_index])):
                
                cur_ob = self.OB(self.cancelations.t[cncl_index], order_books[-1][1].bids, order_books[-1][1].asks)
                cur_ob.process_cancelation(self.cancelations.loc[cncl_index])
                order_books.append((self.cancelations.t[cncl_index], cur_ob))
                cncl_index += 1
                
            elif ((self.limit_orders.t[limit_index] <= self.cancelations.t[cncl_index])
            & (self.limit_orders.t[limit_index] <= self.market_orders.t[market_index])):
                
                cur_ob = self.OB(self.limit_orders.t[limit_index], order_books[-1][1].bids, order_books[-1][1].asks)
                cur_ob.process_limit_order(self.limit_orders.loc[limit_index])
                order_books.append((self.limit_orders.t[limit_index], cur_ob))
                limit_index += 1
                
        self.market_orders.drop(index = self.market_orders.index[-1], axis=0, inplace=True) 
        self.cancelations.drop(index = self.cancelations.index[-1], axis=0, inplace=True) 
        self.limit_orders.drop(index = self.limit_orders.index[-1], axis=0, inplace=True) 
        
        return order_books
                
time = 100
lambda_0 = 20
a = 0.05
var_e = 0.1
q_0 = 10
var_q = 2
b = 0.25
start_mid = 100
depth = 40
step = 0.0025
density = 0.8
lambda_0_limit = 20
lambda_0_cncl = 5
        
sim = Simulator(time, lambda_0, a, var_e, q_0, var_q, b, start_mid, depth, step, density, lambda_0_limit, lambda_0_cncl)
#print(sim.cancelations.loc[5])

#sim.OB_0.process_cancelation(sim.cancelations.loc[5])
#print(sim.OB_O.asks.loc[5])