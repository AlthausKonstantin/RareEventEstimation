#%%
import rareeventestimation as ree
%load_ext autoreload
%autoreload 2
#%%
prob = ree.make_linear_problem(10)
prob.set_sample(2500, seed=2500)
solver = ree.CBREE(seed=1,
             divergence_check=False,
             cvar_tgt=.1,
             tol=0.5,
             num_steps=250,
             save_history=True,
             sigma_inc_max=2,
             observation_window=9,
             return_other=True)
sol2 = solver.solve(prob)

# %%
ree.plot_cbree_parameters(sol2, prob, plot_time=False)
# %%
