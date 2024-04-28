import matplotlib.pyplot as plt
import numpy as np
from brokenaxes import brokenaxes

np.random.seed(123)

x1 = np.random.normal(0, 1, size=1000)
x2 = np.random.normal(-2, 3, size=1000)
x3 = np.random.normal(3, 2.5, size=1000)

fp = open("hotspot-issue-config.config", "r")
fp_readlines = fp.readlines()
x1 = []
for line in fp_readlines:
    if "-trace_issued_sm_id_" in line:
        x1.append(int(line.split(" ")[1].split(",")[0]))
fp.close()
x1 = np.array(x1)


fp = open("b+tree-issue-config.config", "r")
fp_readlines = fp.readlines()
x2 = []
for line in fp_readlines:
    if "-trace_issued_sm_id_" in line:
        x2.append(int(line.split(" ")[1].split(",")[0]))
fp.close()
x2 = np.array(x2)


fp = open("3DConv-issue-config.config", "r")
fp_readlines = fp.readlines()
x3 = []
for line in fp_readlines:
    if "-trace_issued_sm_id_" in line:
        x3.append(int(line.split(" ")[1].split(",")[0]))
fp.close()
x3 = np.array(x3)

print(x3)


kwargs = {
    "bins": 40,
    "histtype": "stepfilled",
    "alpha": 0.5
}

style_list = ['classic']

for style_label in style_list:
    with plt.rc_context({"figure.max_open_warning": len(style_list)}):
        with plt.style.context(style_label):

            fig = plt.figure(num=style_label, figsize=(7.8, 7.8))
            bax = brokenaxes(
                             ylims=((0, 10), (13, 15), (30, 33)),
                             hspace=0.05,
                             despine=False,
                             diag_color='k',
                            )
            bax.hist(x1, ls='none', \
                    color="#c387c3",\
                    label="hotspot", **kwargs)
            bax.hist(x2, ls='none', \
                    color="#fcca99",\
                    label="b+tree", **kwargs)
            bax.hist(x3, ls='none', \
                    color="#8ad9f8",\
                    label="3DConv", **kwargs)
            bax.tick_params(axis='x', labelsize=20)
            bax.tick_params(axis='y', labelsize=20)
            bax.set_xlabel('Total Thread Blocks per SM', fontsize=30, labelpad=25)
            bax.set_ylabel('Number of SMs', fontsize=30)
            bax.legend(loc='best', fontsize=20, frameon=True, shadow=True, \
                      fancybox=False, framealpha=1.0, borderpad=0.3,\
                      ncol=1, markerfirst=True, markerscale=1.3, \
                      numpoints=1, handlelength=2.0)
            bax.grid(True, which='major', axis='both', \
                    linestyle='--', color='gray', linewidth=1)
            plt.show()
            
