import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import minimize


class HA:
    def __init__(self, fun, dim, lb, ub):
        self.fun = fun
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.num_pop = max([100, 20 * dim])
        self.record = []
        self.fun_cnt = 0
        self.gen = 0
        self.niche_num = 3
        self.elite_num = dim
        self.step_size = 1
        self.best = float("inf")
        self.improvement = True

    def ha(self, n):
        pop = np.random.uniform(self.lb, self.ub, (self.num_pop, self.dim))
        fit = np.apply_along_axis(self.fun, 1, pop).reshape(-1, 1)
        self.fun_cnt += self.num_pop
        self.best = min(fit)
        self.record = np.hstack((self.fun_cnt, self.best))
        while self.gen < 20*self.dim:
            self.gen += 1
            print(self.gen)
            pop, fit = self.step_ha(pop, fit)
            # determine this generation is improvement or not
            best_gen = min(fit)
            if best_gen < self.best:
                self.best = best_gen
                self.improvement = True
            else:
                self.improvement = False
        return self.record

    def step_ha(self, pop, fit):
        pop, fit, elite_id = self.clustering_and_learning(pop, fit)
        elite_id += np.argsort(fit, axis=0)[:self.elite_num, 0].tolist()
        # unique and remain original order
        # elite contains niche best and global best
        elite_id = [elite_id[i] for i in sorted(np.unique(elite_id, return_index=True)[1])]
        kid_size = pop.shape[0] - self.elite_num

        offspring = self.inheritance(kid_size, pop, fit)
        mutate_num = round(kid_size * 0.2)
        mutate_id = np.random.choice(kid_size, mutate_num, replace=False)
        offspring[mutate_id, :] = self.mutate(offspring[mutate_id, :])

        # repeat individual only calculate once
        offspring, repeat = np.unique(offspring, axis=0, return_counts=True)
        offspring_fit = np.apply_along_axis(self.fun, 1, offspring).reshape(-1, 1)
        self.fun_cnt += offspring.shape[0]
        repeat -= 1
        repeat_index = np.nonzero(repeat)[0]
        # copy repeat[repeat_index] times offspring[repeat_index, :]
        offspring = np.vstack((offspring, np.repeat(offspring[repeat_index, :], repeat[repeat_index], axis=0)))
        offspring_fit = np.vstack((offspring_fit,
                                   np.repeat(offspring_fit[repeat_index, :], repeat[repeat_index], axis=0)))

        pop = np.vstack((pop[elite_id, :], offspring))
        fit = np.vstack((fit[elite_id, :], offspring_fit))
        self.record = np.vstack((self.record, np.hstack((self.fun_cnt, min(fit)))))
        return pop, fit

    def clustering_and_learning(self, pop, fit):
        elite_id = []
        if self.gen == 1:
            # initial center point randomly
            res = KMeans(n_clusters=self.niche_num, n_init=1).fit(pop)
        else:
            # elites are the center point
            res = KMeans(n_clusters=self.niche_num, init=pop[:self.niche_num, :], n_init=1).fit(pop)
        # get labels of each niche
        labels = res.labels_
        # adaptive learning of elites
        for i in range(self.niche_num):
            idx = np.where(labels == i)[0]
            if len(idx) == 0:
                continue
            best_id = np.argmin(fit[idx, 0])
            # 'L-BFGS-B' with bounds of [lb,ub], maxiter=1
            res = minimize(self.fun, pop[idx[best_id], :], method='L-BFGS-B',
                           bounds=((self.lb, self.ub),) * self.dim,
                           options={'maxiter': 1})
            self.fun_cnt += res.nfev

            # update population and fit
            pop[idx[best_id], :] = res.x
            fit[idx[best_id], 0] = res.fun
            # self.record = np.vstack((self.record, np.hstack((self.fun_cnt, min(fit)))))
            print(self.fun_cnt, min(fit))
            elite_id.append(idx[best_id])
        return pop, fit, elite_id

    def inheritance(self, kid_size, pop, fit):
        id = np.argsort(fit.reshape(-1, ))
        score = fit.copy()
        score[id, :] = np.arange(pop.shape[0], 0, -1).reshape(-1, 1)
        offpsring = np.zeros((kid_size, self.dim))
        mate_size = self.dim
        for i in range(kid_size):
            # randomly choose mate_size individuals from pop
            mate = np.random.choice(pop.shape[0], mate_size, replace=False)
            mate_fit = score[mate]
            mate_pool = pop[mate, :]
            # calculate the probability of each individual based on fitness
            prob = mate_fit / np.sum(mate_fit)
            for j in range(self.dim):
                # choose one gene from mate_pool based on prob
                offpsring[i, j] = np.random.choice(mate_pool[:, j], p=prob.reshape(-1, ))
        return offpsring

    def mutate(self, offspring):
        if self.gen <= 2:
            self.step_size = 1
        else:
            if self.improvement:
                self.step_size = min([1, self.step_size * 4])
            else:
                self.step_size = max([1e-6, self.step_size / 4])
        scale = 1
        # feasible = True
        # constr = True
        tol = 1e-3
        mutation_children = np.zeros((offspring.shape[0], offspring.shape[1]))
        for i in range(offspring.shape[0]):
            x = offspring[i, :].reshape(-1, 1)
            basis, tangent_cone = self.get_directions(self.step_size, x, self.lb, self.ub, tol)
            # if tangent_cone is not empty
            if tangent_cone.shape[1] > 0:
                scale = 1
                # delete the column with all zero
                tangent_cone = tangent_cone[:, np.sum(tangent_cone == 1, axis=0) == 1]

            n_dir_tan = tangent_cone.shape[1]
            n_basis = basis.shape[1]
            # add tangent cone to directions
            dir_vector = np.hstack((basis, tangent_cone))
            # number of directions
            n_dir_total = n_dir_tan + n_basis
            # direction index
            index_vec = np.hstack((np.arange(n_basis), np.arange(n_basis),
                                  np.arange(n_basis, n_dir_total), np.arange(n_basis, n_dir_total)))
            # direction sign
            dir_sign = np.hstack((np.ones(n_basis), -np.ones(n_basis), np.ones(n_dir_tan), -np.ones(n_dir_tan)))

            order_vec = np.random.choice(index_vec.shape[0], index_vec.shape[0], replace=False)
            num_trails = order_vec.shape[0]
            if num_trails > 0:
                for k in range(num_trails):
                    direction = dir_sign[k] * dir_vector[:, index_vec[order_vec[k]]].reshape(-1, 1)
                    mutant = x + self.step_size * scale * direction
                    feasible = self.is_trail_feasible(mutant,self.lb,self.ub,tol)
                    if feasible:
                        mutation_children[i, :] = mutant.reshape(-1, )
                        break
                    else:
                        mutation_children[i, :] = x.reshape(-1, )
            else:
                mutation_children[i, :] = x.reshape(-1, )
        return mutation_children

    def get_directions(self, mesh_size, x, lb, ub, tol):
        lb = np.array([lb] * x.shape[0]).reshape(-1, 1)
        ub = np.array([ub] * x.shape[0]).reshape(-1, 1)
        dim = x.shape[0]
        I = np.eye(dim)
        active = (np.abs(x - lb) < tol) | (np.abs(x - ub) < tol)
        tangent_cone = I[:, active.reshape(-1, )]

        p = 1 / np.sqrt(mesh_size)
        lower_t = np.tril((np.round((p + 1) * np.random.rand(dim, dim) - 0.5)), -1)
        diag_temp = p * np.sign(np.random.rand(dim, 1) - 0.5)
        # make sure diag_temp != 0
        idx = np.where(diag_temp == 0)[0]
        diag_temp[idx, 0] = p * np.sign(0.5 - np.random.rand())
        diag_t = np.diag(diag_temp.reshape(-1, ))
        basis = lower_t + diag_t
        order = np.random.choice(dim, dim, replace=False)
        basis = basis[order][:, order]
        return basis, tangent_cone

    def is_trail_feasible(self,x,lb,ub,tol):
        constraint = 0
        lb = np.array([lb] * x.shape[0]).reshape(-1, 1)
        ub = np.array([ub] * x.shape[0]).reshape(-1, 1)
        constraint = max(max(x - ub), constraint)
        constraint = max(max(lb - x), constraint)
        if constraint < tol:
            return True
        else:
            return False
