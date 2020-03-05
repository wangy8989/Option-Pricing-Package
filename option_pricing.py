# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 13:14:37 2018

@author: Administrator
"""

import numpy as np
from scipy.stats.mstats import gmean
from scipy.stats import norm
from scipy.sparse import coo_matrix
from numpy.linalg import inv


class OptionPricing(object):
    def __init__(self, stock_price, strike_price1, 
                 strike_price2=0.0, strike_price3=0.0):
        self.stock_price = stock_price
        self.strike_price1 = strike_price1
        self.strike_price2 = strike_price2
        self.strike_price3 = strike_price3
        
        
    def European_Call_Payoff(self):
        return np.maximum(self.stock_price-self.strike_price1, 0.0)
    
    def European_Put_Payoff(self):
        return np.maximum(self.strike_price1-self.stock_price, 0.0)
    
    def Bull_Call_Spread(self):
        if not (self.strike_price1 < self.strike_price2):
            print('Warning: inputs are incorrect')
            print('strike_price1 < strike_price2 does not hold')
            quit()
        else:
            call1 = np.maximum(self.stock_price-self.strike_price1, 0.0)
            call2 = np.maximum(self.stock_price-self.strike_price2, 0.0)
            payoff = call1-call2
        return payoff
    
    def Bull_Put_Spread(self):
        if not (self.strike_price1 < self.strike_price2):
            print('Warning: inputs are incorrect')
            print('strike_price1 < strike_price2 does not hold')
            quit()
        else:
            put1 = np.maximum(self.strike_price1-self.stock_price, 0.0)
            put2 = np.maximum(self.strike_price2-self.stock_price, 0.0)
            payoff = put2-put1
        return payoff
    
    def Bear_Call_Spread(self):
        if not (self.strike_price1 < self.strike_price2):
            print('Warning: inputs are incorrect')
            print('strike_price1 < strike_price2 does not hold')
            quit()
        else:
            call1 = np.maximum(self.stock_price-self.strike_price1, 0.0)
            call2 = np.maximum(self.stock_price-self.strike_price2, 0.0)
            payoff = call2-call1
        return payoff
    
    def Collar(self):
        if not (self.strike_price1 < self.strike_price2):
            print('Warning: inputs are incorrect')
            print('strike_price1 < strike_price2 does not hold')
            quit()
        else:
            put = np.maximum(self.strike_price1-self.stock_price, 0.0)
            call = np.maximum(self.stock_price-self.strike_price2, 0.0)
            payoff = put-call
        return payoff
        
    def Straddle(self):
        put = np.maximum(self.strike_price1-self.stock_price, 0.0)
        call = np.maximum(self.stock_price-self.strike_price1, 0.0)
        payoff = put+call
        return payoff
        
    def Strangle(self):
        put = np.maximum(self.strike_price2-self.stock_price, 0.0)
        call = np.maximum(self.stock_price-self.strike_price1, 0.0)
        payoff = put+call
        return payoff
        
    def Butterfly_Spread(self):
        if not ((self.strike_price1 < self.strike_price2) and 
                (self.strike_price2 < self.strike_price3)):
            print('Warning: inputs are incorrect')
            print('strike_price1 < strike_price2 < strike_price3 does not hold')
            quit()
        else:
            lbda = (self.strike_price3-self.strike_price2)/(self.strike_price3
                   -self.strike_price1)
            call1 = np.maximum(self.stock_price-self.strike_price1, 0.0)
            call2 = np.maximum(self.stock_price-self.strike_price2, 0.0)
            call3 = np.maximum(self.stock_price-self.strike_price3, 0.0)
            payoff = lbda*call1 + (1-lbda)*call3 - call2
        return payoff
    
    
    def Black_Scholes_European_Call(self, t, maturity_date, interest_rate, 
                                    dividend_yield, volatility):
        dt = maturity_date-t
        discount = np.exp(-interest_rate*dt)
        y_discount = np.exp(-dividend_yield*dt)
        d1 = ((np.log(self.stock_price/self.strike_price1) + (interest_rate - 
               dividend_yield + 0.5*volatility**2) * dt)
               / (volatility * np.sqrt(dt)))
        d2 = d1 - volatility * np.sqrt(dt)
        
        bs_european_call_price = (self.stock_price * y_discount * norm.cdf(d1) - 
                                  self.strike_price1 * discount * norm.cdf(d2))
        bs_european_call_delta = y_discount * norm.cdf(d1)
        bs_european_call_theta = (- (self.stock_price * norm.pdf(d1) * volatility
                                * y_discount) /(2*np.sqrt(dt)) + dividend_yield 
                                * self.stock_price * norm.cdf(d1) - interest_rate 
                                * self.strike_price1 * discount * norm.cdf(d2))
        
        bs_european_call_vega = (self.stock_price * y_discount * 
                                norm.pdf(d1) * np.sqrt(dt))
        bs_european_call_gamma = y_discount * norm.pdf(d1) / (self.stock_price * 
                                         volatility * np.sqrt(dt))
        bs_european_call_rho = self.strike_price1 * dt * discount * norm.cdf(d2)
        
        return bs_european_call_price, bs_european_call_delta, bs_european_call_theta, bs_european_call_vega, bs_european_call_gamma, bs_european_call_rho
        
        
    def Black_Scholes_European_Put(self, t, maturity_date, interest_rate, 
                                    dividend_yield, volatility):
        dt = maturity_date-t
        discount = np.exp(-interest_rate*dt)
        y_discount = np.exp(-dividend_yield*dt)
        d1 = ((np.log(self.stock_price/self.strike_price1) + (interest_rate - 
               dividend_yield + 0.5*volatility**2) * dt)
               / (volatility * np.sqrt(dt)))
        d2 = d1 - volatility * np.sqrt(dt)
        bs_european_put_price = (self.strike_price1 * discount * norm.cdf(-d2) -
                                 self.stock_price * y_discount * norm.cdf(-d1))
        bs_european_put_delta = - y_discount * (1-norm.cdf(d1))
        bs_european_put_theta = (- (self.stock_price * norm.pdf(d1) * volatility
                                * y_discount) /(2*np.sqrt(dt)) - dividend_yield 
                                * self.stock_price * norm.cdf(-d1) + interest_rate 
                                * self.strike_price1 * discount * norm.cdf(-d2))
        bs_european_put_vega = (self.stock_price * y_discount * 
                                norm.pdf(d1) * np.sqrt(dt))
        bs_european_put_gamma = y_discount * norm.pdf(d1) / (self.stock_price * 
                                         volatility * np.sqrt(dt))
        bs_european_put_rho = -self.strike_price1 * dt * discount * norm.cdf(-d2)

        return bs_european_put_price, bs_european_put_delta, bs_european_put_theta, bs_european_put_vega, bs_european_put_gamma, bs_european_put_rho
         
    
    
    def Black_Scholes_Explicit_FD_EO(self, t, sigma, r, q,
                                     initial_condition, boundary_condition):
        dt = t[1]-t[0]
        ds = self.stock_price[1]-self.stock_price[0]
        
        # check stability condition
        if sigma**2*self.stock_price[-1]**2*dt/(ds**2) > 0.5:
            print('Black Scholes by Explicit Finite Difference is not stable.')
            quit()
            
        else:
            # calculate weight matrix
            t_size = len(t)
            s_size = len(self.stock_price)
            row = np.array([0, s_size-1])
            col = np.array([0, t_size-1])
            data = np.array([1,1])
            sparse_matrix = coo_matrix((data, (row,col)), shape=(s_size, t_size))
            
            for i in range(1,s_size-1):
                col = np.array([i-1,i,i+1])
                row = np.array([i]*3)
                l = 0.5*sigma**2*(self.stock_price[i]/ds)**2*dt - 0.5*(r-q)*(self.stock_price[i]/ds)*dt
                d = 1. - r*dt - sigma**2*(self.stock_price[i]/ds)**2*dt
                u = 0.5*(r-q)*(self.stock_price[i]/ds)*dt + 0.5*sigma**2*(self.stock_price[i]/ds)**2*dt
                data = np.array([l,d,u])
                sparse_matrix += coo_matrix((data, (row, col)), shape=(s_size, t_size))
                
            W = sparse_matrix.toarray() # weight matrix

            
            bs_explicit_fd_eo_price = np.zeros((s_size, t_size)) # value matrix
            
            # initial condition
            if initial_condition == 'ic_call':
                bs_explicit_fd_eo_price[:,0] = self.European_Call_Payoff()
            elif initial_condition == 'ic_put':
                bs_explicit_fd_eo_price[:,0] = self.European_Put_Payoff()
            else:
                print('Not a valid initial condition.')
                quit()
            
            # boundary condition
            if boundary_condition == 'dirichlet_bc':
                if initial_condition == 'ic_call':
                    bs_explicit_fd_eo_price[0,:] = 0
                    for i in range(t_size):
                        bs_explicit_fd_eo_price[-1,i] = (np.exp(-q*dt*i)*self.stock_price[-1] -
                                        self.strike_price1 * np.exp(-r*dt*i))
                else:
                    bs_explicit_fd_eo_price[-1,:] = 0
                    for i in range(t_size):
                        bs_explicit_fd_eo_price[0,i] = (self.strike_price1 * np.exp(-r * dt*i) -
                                                np.exp(-q*dt*i)*self.stock_price[0])
                
            elif boundary_condition == 'neumann_bc':
                # change weight matrix
                l = 0.5*sigma**2*(self.stock_price[0]/ds)**2*dt - 0.5*(r-q)*(self.stock_price[0]/ds)*dt
                d = 1. - r*dt - sigma**2*(self.stock_price[0]/ds)**2*dt
                u = 0.5*(r-q)*(self.stock_price[0]/ds)*dt + 0.5*sigma**2*(self.stock_price[0]/ds)**2*dt
                W[0,:] = np.array([2*l+d, u-l] + [0.]*(t_size-2))
                
                l = 0.5*sigma**2*(self.stock_price[-1]/ds)**2*dt - 0.5*(r-q)*(self.stock_price[-1]/ds)*dt
                d = 1. - r*dt - sigma**2*(self.stock_price[-1]/ds)**2*dt
                u = 0.5*(r-q)*(self.stock_price[-1]/ds)*dt + 0.5*sigma**2*(self.stock_price[-1]/ds)**2*dt
                W[-1,:] = np.array([0.]*(t_size-2) + [l-u,d+2*u])

            else:
                print('Not a valid boundary condition.')
                quit()
            
            # calculate value matrix
            for j in range(0,t_size-1):
                bs_explicit_fd_eo_price[:,j+1] = W @ bs_explicit_fd_eo_price[:,j]

        
        return bs_explicit_fd_eo_price
        
    
    
    def Black_Scholes_Implicit_FD_EO(self, t, sigma, r, q,
                                     initial_condition, boundary_condition):
        dt = t[1]-t[0]
        ds = self.stock_price[1]-self.stock_price[0]
        
        # calculate weight matrix
        t_size = len(t)
        s_size = len(self.stock_price)
        row = np.array([0, s_size-1])
        col = np.array([0, t_size-1])
        data = np.array([1,1])
        sparse_matrix = coo_matrix((data, (row,col)), shape=(s_size, t_size))
        
        for i in range(1,s_size-1):
            col = np.array([i-1,i,i+1])
            row = np.array([i]*3)
            l = 0.5*(r-q)*(self.stock_price[i]/ds)*dt - 0.5*sigma**2*(self.stock_price[i]/ds)**2*dt
            d = 1. + r*dt + sigma**2*(self.stock_price[i]/ds)**2*dt
            u = - 0.5*(r-q)*(self.stock_price[i]/ds)*dt - 0.5*sigma**2*(self.stock_price[i]/ds)**2*dt
            data = np.array([l,d,u])
            sparse_matrix += coo_matrix((data, (row, col)), shape=(s_size, t_size))
        W = sparse_matrix.toarray() # weight matrix
        
        bs_implicit_fd_eo_price = np.zeros((s_size, t_size)) # value matrix
        
        # initial condition
        if initial_condition == 'ic_call':
            bs_implicit_fd_eo_price[:,0] = self.European_Call_Payoff()
        elif initial_condition == 'ic_put':
            bs_implicit_fd_eo_price[:,0] = self.European_Put_Payoff()
        else:
            print('Not a valid initial condition.')
            quit()
        
        # boundary condition
        if boundary_condition == 'dirichlet_bc':
            if initial_condition == 'ic_call':
                bs_implicit_fd_eo_price[0,:] = 0
                for i in range(t_size):
                    bs_implicit_fd_eo_price[-1,i] = (np.exp(-q*dt*i)*self.stock_price[-1] -
                                    self.strike_price1 * np.exp(-r*dt*i))
            else:
                bs_implicit_fd_eo_price[-1,:] = 0
                for i in range(t_size):
                    bs_implicit_fd_eo_price[0,i] = (self.strike_price1 * np.exp(-r * dt*i)
                                        - np.exp(-q*dt*i)*self.stock_price[0])
            
        elif boundary_condition == 'neumann_bc':
            # change weight matrix
            l = 0.5*(r-q)*(self.stock_price[0]/ds)*dt - 0.5*sigma**2*(self.stock_price[0]/ds)**2*dt
            d = 1. + r*dt + sigma**2*(self.stock_price[0]/ds)**2*dt
            u = - 0.5*(r-q)*(self.stock_price[0]/ds)*dt - 0.5*sigma**2*(self.stock_price[0]/ds)**2*dt
            W[0,:] = np.array([2*l+d, u-l] + [0.]*(t_size-2))
            
            l = 0.5*(r-q)*(self.stock_price[-1]/ds)*dt - 0.5*sigma**2*(self.stock_price[-1]/ds)**2*dt
            d = 1. + r*dt + sigma**2*(self.stock_price[-1]/ds)**2*dt
            u = - 0.5*(r-q)*(self.stock_price[-1]/ds)*dt - 0.5*sigma**2*(self.stock_price[-1]/ds)**2*dt
            W[-1,:] = np.array([0.]*(t_size-2) + [l-u,d+2*u])
            
        else:
            print('Not a valid boundary condition.')
            quit()
        
        # calculate value matrix
        for j in range(t_size-1):
            bs_implicit_fd_eo_price[:,j+1] = np.linalg.solve(W, 
                                   bs_implicit_fd_eo_price[:,j])
        
        return bs_implicit_fd_eo_price
    
    
    
    def Black_Scholes_Theta_Scheme_FD_EO(self, t, sigma, r, q, theta,
                                     initial_condition, boundary_condition):
        dt = t[1]-t[0]
#        ds = self.stock_price[1]-self.stock_price[0]
        
        # calculate weight matrix for implicit and explicit
        t_size = len(t)
        s_size = len(self.stock_price)
        row = np.array([0, s_size-1])
        col = np.array([0, t_size-1])
        data = np.array([1, 1])
        sparse_matrix1 = coo_matrix((data, (row,col)), shape=(s_size, t_size))
        sparse_matrix2 = coo_matrix((data, (row,col)), shape=(s_size, t_size))
        
        for i in range(1,s_size-1):
            ds = (self.stock_price[i+1] - self.stock_price[i-1])/2
            col = np.array([i-1,i,i+1])
            row = np.array([i]*3)
            
            l = 0.5*(r-q)*(self.stock_price[i]/ds)*dt - 0.5*sigma**2*(self.stock_price[i]/ds)**2*dt
            d = 1. + r*dt + sigma**2*(self.stock_price[i]/ds)**2*dt
            u = - 0.5*(r-q)*(self.stock_price[i]/ds)*dt - 0.5*sigma**2*(self.stock_price[i]/ds)**2*dt
            data = np.array([l,d,u])
            sparse_matrix1 += coo_matrix((data, (row, col)), shape=(s_size, t_size))
            
            l = 0.5*sigma**2*(self.stock_price[i]/ds)**2*dt - 0.5*(r-q)*(self.stock_price[i]/ds)*dt
            d = 1. - r*dt - sigma**2*(self.stock_price[i]/ds)**2*dt
            u = 0.5*(r-q)*(self.stock_price[i]/ds)*dt + 0.5*sigma**2*(self.stock_price[i]/ds)**2*dt
            data = np.array([l,d,u])
            sparse_matrix2 += coo_matrix((data, (row, col)), shape=(s_size, t_size))

            
        W1 = sparse_matrix1.toarray() # weight matrix for implicit
        W2 = sparse_matrix2.toarray() # weight matrix for explicit

        bs_theta_scheme_fd_eo_price = np.zeros((s_size, t_size)) # value matrix
        
        # initial condition
        if initial_condition == 'ic_call':
            bs_theta_scheme_fd_eo_price[:,0] = self.European_Call_Payoff()
        elif initial_condition == 'ic_put':
            bs_theta_scheme_fd_eo_price[:,0] = self.European_Put_Payoff()
        else:
            print('Not a valid initial condition.')
            quit()
        
        # boundary condition
        if boundary_condition == 'dirichlet_bc':
            if initial_condition == 'ic_call':
                bs_theta_scheme_fd_eo_price[0,:] = 0
                for i in range(t_size):
                    bs_theta_scheme_fd_eo_price[-1,i] = (np.exp(-q*dt*i)*self.stock_price[-1] -
                                    self.strike_price1 * np.exp(-r*dt*i))
            else:
                bs_theta_scheme_fd_eo_price[-1,:] = 0
                for i in range(t_size):
                    bs_theta_scheme_fd_eo_price[0,i] = (self.strike_price1 * np.exp(-r * dt*i)
                                        - np.exp(-q*dt*i)*self.stock_price[0])
#            print(bs_theta_scheme_fd_eo_price[-1,:])  
        
        elif boundary_condition == 'neumann_bc':
            ds = self.stock_price[1] - self.stock_price[0]
            # change weight matrix
            # implicit
            i_l = 0.5*(r-q)*(self.stock_price[0]/ds)*dt - 0.5*sigma**2*(self.stock_price[0]/ds)**2*dt
            i_d = 1. + r*dt + sigma**2*(self.stock_price[0]/ds)**2*dt
            i_u = - 0.5*(r-q)*(self.stock_price[0]/ds)*dt - 0.5*sigma**2*(self.stock_price[0]/ds)**2*dt
            # explicit
            e_l = 0.5*sigma**2*(self.stock_price[0]/ds)**2*dt - 0.5*(r-q)*(self.stock_price[0]/ds)*dt
            e_d = 1. - r*dt - sigma**2*(self.stock_price[0]/ds)**2*dt
            e_u = 0.5*(r-q)*(self.stock_price[0]/ds)*dt + 0.5*sigma**2*(self.stock_price[0]/ds)**2*dt

            W1[0,:] = np.array([2*i_l+i_d, i_u-i_l] + [0.]*(t_size-2))
            W2[0,:] = np.array([2*e_l+e_d, e_u-e_l] + [0.]*(t_size-2))
            
            ds = self.stock_price[-1] - self.stock_price[-2]
            i_l = 0.5*(r-q)*(self.stock_price[-1]/ds)*dt - 0.5*sigma**2*(self.stock_price[-1]/ds)**2*dt
            i_d = 1. + r*dt + sigma**2*(self.stock_price[-1]/ds)**2*dt
            i_u = - 0.5*(r-q)*(self.stock_price[-1]/ds)*dt - 0.5*sigma**2*(self.stock_price[-1]/ds)**2*dt
            e_l = 0.5*sigma**2*(self.stock_price[-1]/ds)**2*dt - 0.5*(r-q)*(self.stock_price[-1]/ds)*dt
            e_d = 1. - r*dt - sigma**2*(self.stock_price[-1]/ds)**2*dt
            e_u = 0.5*(r-q)*(self.stock_price[-1]/ds)*dt + 0.5*sigma**2*(self.stock_price[-1]/ds)**2*dt

            W1[-1,:] = np.array([0.]*(t_size-2) + [i_l-i_u, i_d+2*i_u])
            W2[-1,:] = np.array([0.]*(t_size-2) + [e_l-e_u, e_d+2*e_u])
            
        else:
            print('Not a valid boundary condition.')
            quit()
        
        
        # calculate value matrix
        for j in range(t_size-1):
            identity = np.identity(t_size)
            bs_theta_scheme_fd_eo_price[:,j+1] = np.linalg.solve((theta*W1+(1-theta)*identity), 
                                   (((1-theta)*W2+theta*identity) @ bs_theta_scheme_fd_eo_price[:,j]))
            if initial_condition == 'ic_call':
                bs_theta_scheme_fd_eo_price[-1,j+1] = (np.exp(-q*dt*(j+1))*self.stock_price[-1] -
                                    self.strike_price1 * np.exp(-r*dt*(j+1)))
            else:
                bs_theta_scheme_fd_eo_price[0,j+1] = (self.strike_price1 * np.exp(-r * dt*(j+1))
                                        - np.exp(-q*dt*(j+1))*self.stock_price[0])
        
        return bs_theta_scheme_fd_eo_price
    

    def Gaussian_RBF(self, epsilon, x, centre_note):
        r = (x - centre_note) ** 2
        phi_ga_rbf = np.exp(-epsilon**2*r)
        phi_x_ga_rbf = -2*(epsilon**2)*(x-centre_note)*phi_ga_rbf
        phi_xx_ga_rbf = 4*(epsilon**4)*(x-centre_note)**2*phi_ga_rbf - 2*epsilon**2*phi_ga_rbf
        return phi_ga_rbf, phi_x_ga_rbf, phi_xx_ga_rbf

        
    def Multiquadric_RBF(self, epsilon, x, centre_note):
        phi_mq_rbf = np.sqrt(1 + (epsilon**2) * ((x - centre_note)**2))
        phi_x_mq_rbf = epsilon**2*(x-centre_note)/phi_mq_rbf
        phi_xx_mq_rbf = (epsilon**2/phi_mq_rbf
                - epsilon**4*(x-centre_note)**2/(epsilon**2*(x-centre_note)**2+1)**3)
        return phi_mq_rbf, phi_x_mq_rbf, phi_xx_mq_rbf
        
        
    def Inverse_Multiquadric_RBF(self, epsilon, x, centre_note):
        phi_imq_rbf = 1/np.sqrt(1 + (epsilon**2) * ((x - centre_note)**2))
        phi_x_imq_rbf = -epsilon**2*(x-centre_note)/np.sqrt((1+epsilon**2*(x-centre_note)**2)**3)
        phi_xx_imq_rbf = (-epsilon**2/np.sqrt((1+epsilon**2*(x-centre_note)**2)**3) 
                + (3*epsilon**4*(x-centre_note)**2)/np.sqrt((1+epsilon**2*(x-centre_note)**2)**5))
        return phi_imq_rbf, phi_x_imq_rbf, phi_xx_imq_rbf
        
        
    def Inverse_Quadric_RBF(self, epsilon, x, centre_note):
        phi_iq_rbf = 1/(1 + (epsilon**2) * ((x - centre_note)**2))
        phi_x_iq_rbf = -2*epsilon**2*(x-centre_note)/((1+epsilon**2*(x-centre_note)**2)**2)
        phi_xx_iq_rbf = (-2*epsilon**2/((1+epsilon**2*(x-centre_note)**2)**2) 
                + 8*epsilon**4*(x-centre_note)**2/(1+epsilon**2*(x-centre_note)**2)**3)
        return phi_iq_rbf, phi_x_iq_rbf, phi_xx_iq_rbf

    
    # x is evenly spaced according to log prices
    def Black_Scholes_Global_RBF_EO(self, t, maturity_date, m, n, s_max, s_min, interest_rate, dividend_yield, volatility, 
                                    initial_condition, boundary_condition, rbf_function):     
        dt = 1/m
        x_max=np.log(s_max)
        x_min=np.log(s_min)
        x = np.linspace(x_min,x_max,n)  
        
        tao = np.arange(0,m)*dt
        lmd = np.zeros((m,n))   #lambda
        bs_global_rbf_eo_price = np.zeros((n,m))
        
        L, Lx, Lxx = rbf_function
#        print(L)
        
        Linv = inv(L)
        P = interest_rate*np.identity(n) - (interest_rate-0.5*volatility**2)*Linv.dot(Lx) - 0.5*volatility**2*Linv.dot(Lxx)

        if initial_condition ==  'ic_call':
            #initial condition
            bs_global_rbf_eo_price[:,0] = np.maximum(np.exp(x)-self.strike_price1,0)
            bs_global_rbf_eo_price[0,0] = 0
            bs_global_rbf_eo_price[-1,0] = s_max - self.strike_price1
            
            #lambda
            lmd[0] = np.linalg.solve(L, bs_global_rbf_eo_price[:,0])
    
            for i in range(1,m):
                lmd[i] = (inv(np.identity(n)+0.5*dt*P).dot(np.identity(n)-0.5*dt*P)).dot(lmd[i-1])
                bs_global_rbf_eo_price[:,i] = np.dot(L,lmd[i])     
                #update boundary
                bs_global_rbf_eo_price[0][i] = 0
                bs_global_rbf_eo_price[-1][i] = s_max - self.strike_price1*np.exp(-interest_rate*tao[i]) 
                #update lambda
                lmd[i] = Linv.dot(bs_global_rbf_eo_price[:,i])

        elif initial_condition ==  'ic_put':
            #initial condition
            bs_global_rbf_eo_price[:,0] = np.maximum(self.strike_price1-np.exp(x),0)
            bs_global_rbf_eo_price[0,0] = self.strike_price1 - s_min
            bs_global_rbf_eo_price[-1,0] = 0
            
            #lambda
            lmd[0] = Linv.dot(bs_global_rbf_eo_price[:,0])
    

            for i in range(1,m):
                lmd[i] = inv(np.identity(n)+0.5*dt*P).dot(np.identity(n)-0.5*dt*P).dot(lmd[i-1])
                bs_global_rbf_eo_price[:,i] = L.dot(lmd[i])       
                #update boundary
                bs_global_rbf_eo_price[0][i] = self.strike_price1*np.exp(-interest_rate*tao[i]) - s_min
                bs_global_rbf_eo_price[-1][i] = 0       
                #update lambda
                lmd[i] = Linv.dot(bs_global_rbf_eo_price[:,i])
                
        else:
            print ('Not a valid initial condition')
            quit()

        return bs_global_rbf_eo_price
  

    def Black_Scholes_RBF_FD_EO(self, t, maturity_date, m, n, s_max, s_min, interest_rate, 
                                dividend_yield, volatility, initial_condition, 
                                boundary_condition, rbf_function):
        dt = 1/m        
        x = np.log(np.linspace(s_min,s_max,n))
        
        tao = np.arange(0,m)*dt
        bs_rbf_fd_eo_price = np.zeros((n,m))

        L, Lx, Lxx = rbf_function

        #construct weighted matrix
        W = np.zeros ((n-2,n-2))
        D = (interest_rate-0.5*volatility**2)*Lx + 0.5*volatility**2*Lxx - interest_rate*L
        for i in range(1,n-1):
            WW = inv(L[(i-1):(i+2),(i-1):(i+2)]).dot(D[(i-1):(i+2),i].reshape(3,1)).flatten()          
            if i==1:
                w1 = WW.copy()
                W[i-1][(i-1):(i+1)] = WW[1:].copy()
            elif i==m-2:
                wm = WW.copy()
                W[i-1][(i-2):i] = WW[0:2].copy()
            else:
                W[i-1][(i-2):(i+1)] = WW.copy()            
            
        
        if initial_condition ==  'ic_call':
            bs_rbf_fd_eo_price[:,0] = np.maximum(np.exp(x)-self.strike_price1,0)
            bs_rbf_fd_eo_price[0,:] = 0
            bs_rbf_fd_eo_price[-1,:] = s_max - self.strike_price1*np.exp(-interest_rate*tao)

        elif initial_condition ==  'ic_put':
            bs_rbf_fd_eo_price[:,0] = np.maximum(self.strike_price1-np.exp(x),0)
            bs_rbf_fd_eo_price[0,:] = self.strike_price1*np.exp(-interest_rate*tao) - s_min
            bs_rbf_fd_eo_price[-1][0] = 0
            
        else:
            print ("Not a valid initial condition")
            quit()


        Iinv = inv(np.identity(n-2)-0.5*dt*W)
        I =np.identity(n-2) + 0.5*dt*W
        v = np.zeros(n-2)
        
        for i in range(m-1):
            # set boundary condition
            v[0] = (bs_rbf_fd_eo_price[0,i+1] + bs_rbf_fd_eo_price[0,i]) * 0.5*dt*w1[0]
            v[-1] = (bs_rbf_fd_eo_price[-1,i+1] + bs_rbf_fd_eo_price[-1,i]) * 0.5*dt*wm[2]
            bs_rbf_fd_eo_price[1:-1,i+1] = Iinv.dot(I.dot(bs_rbf_fd_eo_price[1:-1,i])+v)
        
        return bs_rbf_fd_eo_price

    
    # for one trajectory or many
    def Geometric_Brownian_Motion_Trajectory(self, mu, q, sigma, t, trials=1):
        path = np.zeros(shape=(trials,len(t)))
        path[:,0] = self.stock_price
        for i in range(len(t)-1):
            dt = t[i+1]-t[i]
            dw = np.random.normal(loc=0,scale=np.sqrt(dt),size=trials)
            path[:,i+1] = path[:,i] * np.exp(((mu-q)-0.5*sigma**2)*dt+sigma*dw)
        return path
    
    # GBM with jump
    # y is lognormal(a,b^2)
    def Geometric_Brownian_Motion_Jump(self, mu, q, sigma, t, a, b, lmda, trials=1):
        
        path = np.zeros(shape=(trials,len(t)))
        path[:,0] = np.log(self.stock_price)
        for i in range(len(t)-1):
            dt = t[i+1]-t[i]
            dw1 = np.random.normal(loc=0,scale=np.sqrt(dt),size=trials)
            n = np.random.poisson(lam=lmda*dt,size=trials)
            if n == 0:
                jump = 0
            else:
                dw2 = np.random.normal(loc=0,scale=1,size=trials)
                jump = a*n + b*np.sqrt(n)*dw2
                
            path[:,i+1] = path[:,i] + ((mu-q)-0.5*sigma**2)*dt + sigma*dw1 + jump
            
        return path[0]
    
    # s_path is a path of stock price
    def Arithmetic_Average_Price_Asian_Call(self, s_path, T, r):
        average = np.mean(s_path) # mean of every row (every path)
        arithm_asian = np.exp(-r*T) * np.maximum(average-self.strike_price1, 0)
        return arithm_asian
    
    def Geometric_Average_Price_Asian_Call(self, s_path, T, r):
        average = gmean(s_path) # mean of every row (every path)
        geo_asian = np.exp(-r*T) * np.maximum(average-self.strike_price1, 0)
        return geo_asian
    
    def BS_Geometric_Average_Price_Asian_Call(self, sigma, q, r, T):
        sig = sigma/np.sqrt(3)
        b = 0.5*(r+q+sigma**2/6)
        d1 = ((np.log(self.stock_price/self.strike_price1) + 
              (r-b+0.5*sig**2)*T)/sig*np.sqrt(T))
        d2 = d1 - sig*np.sqrt(T)
        geo_asian = (self.stock_price*np.exp(-b*T)*norm.cdf(d1) - 
                     self.strike_price1*np.exp(-r*T)*norm.cdf(d2))
        return geo_asian
    
    # x_bar is geo_asian by BS
    # x is geo asian by monte carlo
    # y is arithmetic asian by monte carlo
    # approximate arithm asian
    def Control_Variates_Arithmetic_Average_Asian_Call(self, x, y, x_BS):
        b = np.cov(x,y)[0,1]/np.var(x)
#        print(np.cov(x,y))
        cv_arithm_asian = y - b*(x-x_BS)
        return cv_arithm_asian
        
        