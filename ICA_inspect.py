import ICAize
import stack
import matplotlib.pyplot as plt
import numpy as np

fastica = ICAize.unpickle_FastICA(target_type="combined")
for comp_i in range(min(fastica.components_.shape[0], 10)):
    scale_factor = 2.4/np.max(np.abs(fastica.components_[comp_i]))
    plt.plot(stack.skyexp_wlen_out, (fastica.components_[comp_i]*scale_factor)+(5*comp_i) )

plt.show()
plt.close()


fastica = ICAize.unpickle_SPCA(target_type="combined")
for comp_i in range(min(fastica.components_.shape[0], 10)):
    scale_factor = 2.4/np.max(np.abs(fastica.components_[comp_i]))
    plt.plot(stack.skyexp_wlen_out, (fastica.components_[comp_i]*scale_factor)+(5*comp_i) )

plt.show()
plt.close()
