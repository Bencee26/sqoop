from matplotlib import pyplot as plt
import os
import csv
import operator

path = 'saved/'
error_tuple = []
for folder in os.listdir(path):

    if "FiLM" in folder:
        if str.isdigit(folder[-2]):
            k = int(folder[-2:])
        else:
            k = int(folder[-1])
        with open(path+folder+'/test_stats.csv', 'r') as csvfile:
            fieldnames = ['epoch', 'num_iters', 'accuracy']
            reader = csv.DictReader(csvfile, fieldnames)
            all_lines = list(reader)
            last_line = all_lines[-1]
            err = 100*(1-float(last_line['accuracy']))
        error_tuple.append((k, err))

error_tuple.sort(key = operator.itemgetter(0))
ks = [str(tup[0]) for tup in error_tuple]
error = [tup[1] for tup in error_tuple]
fig, ax = plt.subplots()
rects = ax.bar(ks, error, width=0.5, color=['y', 'g', 'b', 'm', 'c', 'r'])
ax.set_title('Error rate with FiLM architecture')
ax.set_ylabel('error rate on test set')
ax.set_xlabel('number of right hand side characters while training')
for rect in rects:
    height = rect.get_height()
    ax.text(rect.get_x()+0.1, height + 0.2, str(round(height, 2)))
plt.show()