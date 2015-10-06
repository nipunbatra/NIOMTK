run energy_square_area_nilm.py
run energy_square_area_nilm.py
plot_scatter()
run energy_square_area_nilm.py
plot_scatter()
run energy_square_area_nilm.py
plot_scatter()
run energy_square_area_nilm.py
plot_scatter()
df_nilm
df_nilm.columns
run energy_square_area_nilm.py
plot_scatter()
results
df_nilm
%paste
df_result
df_result.corr()
df_nilm
df_nilm.pred_hvac_max
df_nilm.head()
df.head()
df.pred_hvac_max
df.pred_hvac_max.head()
df_nilm.pred_hvac_max
df_nilm.pred_hvac_max.head()
df.pred_hvac_max.head()
run find_ac_power.py
out
%paste
pred_df_fhmm
%paste
pred_df_ac_fhmm
pred_ser_ac = pred_df_ac_fhmm.squeeze()
pred_ser_ac
pred_ser_ac.name = "FHMM"
%paste
a = _
a
a.sum()
%paste
out
out = {}
%paste
%paste
%paste
feature
%paste
out
pd.DataFrame(out)
pd.DataFrame(out[26])
pd.DataFrame(out[26]).astype('int')
store2.keys()
store2['/26']
%paste
out.keys()
out[9934]
out[9934]['Hart']
pd.HDFStore("/Users/nipunbatra/git/NIOMTK/class.h5")['/nilm']
out[9934]
hart_dict = {}
fhmm_dict = {}
fpr key, v in out.iteritems():
for k, v in out.iteritems():
    fhmm_dict[k] = v['FHMM']
    hart_dict[k] = v['Hart']
fhmm_dict
pd.DataFrame(fhmm_dict)
pd.DataFrame(fhmm_dict).astype('int')
pd.DataFrame(fhmm_dict).dropna()
pd.DataFrame(fhmm_dict).T
pd.DataFrame(fhmm_dict).T.astype('float')
pd.DataFrame(fhmm_dict).T.astype('float')
sore
store
store2.keys()
store2['/FHMM'] = pd.DataFrame(fhmm_dict).T.astype('float')
store2['/Hart'] = pd.DataFrame(hart_dict).T.astype('float')
%paste
k
store2['/1953']
%paste
alpha
pd.DataFrame(alpha)
pd.DataFrame(alpha["Hart"])
pd.DataFrame(alpha["Hart"]).T
pd.DataFrame(alpha["Hart"]).T.describe()
pd.DataFrame(alpha["FHMM"]).T.describe()
pd.DataFrame(alpha["FHMM"]).T.mean()
pd.DataFrame(alpha["FHMM"]).T.mean().T
run energy_square_area_nilm_multiple_algorithms.py
run energy_square_area_nilm_multiple_algorithms.py
run energy_square_area_nilm_multiple_algorithms.py
df
store.keys()
run energy_square_area_nilm_multiple_algorithms.py
df_hart
df_fhmm
plot_classification_nilm(NMAX=1)
df_nilm
df_nilm.isnan()
df_nilm.isnull()
df_nilm.isnull().sum()
df_hart
df_hart.dropna()
run energy_square_area_nilm_multiple_algorithms.py
plot_classification_nilm(NMAX=1)
run energy_square_area_nilm_multiple_algorithms.py
plot_classification_nilm(NMAX=1)
df_hart
df_fhmm
run energy_square_area_nilm_multiple_algorithms.py
plot_classification_nilm(NMAX=1)
run energy_square_area_nilm_multiple_algorithms.py
plot_classification_nilm(NMAX=1)
run energy_square_area_nilm_multiple_algorithms.py
plot_classification_nilm(NMAX=1)
plot_classification_nilm(NMAX=10)
temp2 = temp.copy()
temp = plot_classification_nilm(NMAX=10)
temp = _117
temp
run energy_square_area.py
a, b, c = plot_classification(10)
a
temp
new_df = [[74.3, 62.0, 48.0, 68], [73.6, 61.2, 48.0, 77.6],
    [77.27, 64.9, 56.0, 77.6], [78.5, 70.9, 50.0, 77.6]]
new_df = pd.DataFrame([[74.3, 62.0, 48.0, 68], [73.6, 61.2, 48.0, 77.6], [
                      77.27, 64.9, 56.0, 77.6], [78.5, 70.9, 50.0, 77.6]])
new_df
new_df.T
new_df = new_df.T
new_df
new_df.columns = ["Only Aggregate", "FHMM", "Hart", "Submetered"]]
new_df.columns = ["Only Aggregate", "FHMM", "Hart", "Submetered"]
new_df
new_df.index = a.index[2: ]
new_df
latexify(columns=1)
import sys
sys.path.append("../common")
    from common_functions import latexify, format_axes
latexify(columns=1)
ax = new_df.plot(kind="bar", rot=0)
% matplotlib qt
ax = new_df.plot(kind="bar", rot=0)
ax = new_df.plot(kind="bar", rot=0)
ax.legend(pos=1)
ax.legend(loc=1)
ax.legend(loc=2)
plt.draw()
plt.draw(loc=3)
ax.legend(loc=3)
plt.draw()
ax.legend(loc=4)
plt.draw()
format_axes(ax)
temp
a
ax.set_ylabel("Accuracy")
plt.savefig(
    "/Users/nipunbatra/git/NIOMTK/figures/class_nilm.pdf", bbox_inches = "tight")
df
df_hart
df_hart["night_mean_power"]
df_hart.columns
a=pd.DataFrame(
    {"hart": df_hart["hvac_night_mean"], "fhmm": df["hvac_night_mean"]})
a
a.corr()
a.astype('float')
a.astype('float').corr()
a=pd.DataFrame(
    {"hart": df_fhmm["hvac_night_mean"], "fhmm": df["hvac_night_mean"]})
a.astype('float').corr()
temp
a
df
