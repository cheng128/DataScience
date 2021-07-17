# you must use python 3.6, 3.7, 3.8, 3.9 for sourcedefender
import sourcedefender
import numpy as np
from HomeworkFramework import Function
from math import floor, log, sqrt
from cmath import sqrt as csqrt


class RS_optimizer(Function): # need to inherit this class "Function"
    def __init__(self, target_func):
        super().__init__(target_func) # must have this init to work normally

        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)

        self.target_func = target_func

        self.eval_times = 0
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)
        

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def run(self, FES): # main part for your implementation
        """
        --------------------- Initialization ------------------------
        """
        # inpute parameters
        N = self.dim
        xmean = np.mat(np.random.uniform(0, 1, N)).T           # objective variables initial point
        while 0 in xmean:
            xmean = np.mat(np.random.uniform(0, 1, N)).T
        sigma = (self.upper - self.lower) * 0.3      # step-size
        
        # strategy parameter setting: Selection
        offspr_num = 4 + floor(3 * np.log(N))
        mu = offspr_num / 2
        weights = np.mat([log(mu + 1 / 2)]) - np.mat([log(i) for i in range(1, floor(mu) + 1)]).H
        mu = floor(mu)
        weights = weights / sum(weights)
        mueff = np.sum(weights) ** 2 / np.sum([w ** 2 for w in weights])
        
        # strategy parameter setting: Adaption
        cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
        cs = (mueff + 2) / (N + mueff + 5)
        cone = 2 / ((N + 1.3) ** 2 + mueff)
        cmu = 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + 2 * mueff / 2)
        damps = 1 + 2 * max(0, sqrt((mueff - 1) / (N + 1)) - 1) + cs

        # Initialize dynamic (internal) strategy parameters and constants
        pc, ps = np.mat([float(0)] * N).T, np.mat([float(0)] * N).T
        B, D = np.mat(np.eye(N)), np.mat(np.eye(N))
        BD = B.dot(D)
        C = BD.dot(BD.H)             # B * D * ( B * D )'
        eigeneval = 0
        chiN = N ** 0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))
        
        """
        --------------------- Generation Loop ------------------------
        """
        
        ReachFunctionLimit = False
        # while counteval < stopeval
        while self.eval_times < FES:
            print('=====================FE=====================')
            print(self.eval_times)
            arz = np.zeros(N * offspr_num).reshape(N, offspr_num)
            arx = np.zeros(N * offspr_num).reshape(N, offspr_num)
            normal = np.zeros(N * offspr_num).reshape(N, offspr_num)
            value = [float(0)] * offspr_num
            # Generate and evaluate lambda offspring
            for k in range(offspr_num):
                arz[:, k] = np.random.normal(0, 1, N)
                arx[:, k] = (xmean + sigma * (B * D * arz[:, k].transpose().reshape(N,1))).transpose()
                for idx, num in enumerate(arx[:, k]):
                    if num > self.upper:
                        arx[:, k][idx] = self.upper
                    elif num < self.lower:
                        arx[:, k][idx] = self.lower
                        
                result = self.f.evaluate(func_num, arx[:, k])
                if result == 'ReachFunctionLimit':
                    ReachFunctionLimit = True
                    self.eval_times += 1
                    break
                else:
                    value[k] = result
                    self.eval_times += 1

            # Sort by fitness and compute weighted mean into xmean
            arfit_idx = sorted([(arfit, idx) for idx, arfit in enumerate(value)], key=lambda x: x[0])
            arfitness = [sol[0] for sol in arfit_idx]
            arindex = [sol[1] for sol in arfit_idx]
            xmean = arx[:, arindex[:mu]] * weights
            zmean = arz[:, arindex[:mu]] * weights

            # Cumulation: Update evolution paths
            ps = (1 - cs) * ps + (sqrt(cs * (2 - cs) * mueff)) * (B * zmean)

            if np.linalg.norm(ps) / sqrt(1 - (1 - cs) ** (2 * self.eval_times/offspr_num)) / chiN < 1.4 + 2 / (N + 1):
                hsig = 1
            else:
                hsig = 0

            pc = (1 - cc) * pc + hsig * sqrt(cc * (2 - cc) * mueff) * (B * D * zmean)

            # Adapt covariance matrix C
            C = (1 - cone - cmu) * C \
                + cone * (pc * pc.H + (1 - hsig) * cc * (2 - cc) * C) \
                + cmu * (B * D * arz[:, arindex[:mu]]) * np.diagflat(weights) * (B * D * arz[:, arindex[:mu]]).H

            # Adapt step-size sigma
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

            # Update B and D from C
            if (self.eval_times - eigeneval) > (offspr_num / (cone + cmu) / N / 10):               
                eigeneval = self.eval_times
                C = np.mat(np.triu(C)) + np.mat(np.triu(C, 1)).H
                D, B = np.linalg.eigh(C)
                D = np.diagflat([csqrt(i) for i in D])

            if ReachFunctionLimit:
                print("out ReachFunctionLimit")
                break 
                
            if float(value[0]) < self.optimal_value:
                self.optimal_solution[:] = np.array(arx[:, arindex[0]])
                self.optimal_value = float(value[0])

            print("optimal: {}\n".format(self.get_optimal()[1]))
            

if __name__ == '__main__':
    func_num = 1
    fes = 0
    #function1: 1000, function2: 1500, function3: 2000, function4: 2500
    while func_num < 5:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000 
        else:
            fes = 2500

        # you should implement your optimizer
        op = RS_optimizer(func_num)
        op.run(fes)
        
        best_input, best_value = op.get_optimal()
        print(best_input, best_value)
        
        # change the name of this file to your student_ID and it will output properlly
        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1 
