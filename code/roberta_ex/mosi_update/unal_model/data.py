'''import pickle
mosi_data=pickle.load(open('D:\\data_multimoda\\mosi\\unaligned_50.pkl','rb'))
print(mosi_data.keys())
print(mosi_data['train'].keys())
print(mosi_data['train']['audio'][0].shape)
print(mosi_data['train']['vision'][0].shape)
print(mosi_data['train']['classification_labels'][0])
print(mosi_data['train']['regression_labels'][0])
for i in range(len(mosi_data['train']['audio'])):
    print(mosi_data['train']['regression_labels'][i],mosi_data['train']['regression_labels'][i],mosi_data['train']['annotations'][i])'''
import numpy
predict_list=[-1,-2,0,1,2,0,3]
truth_list=[0,1,2,-1,-2,-3,-1]
exclude_zero = True
non_zeros = numpy.array([i for i, e in enumerate(truth_list) if e != 0 or (not exclude_zero)])
predict_list = numpy.array(predict_list).reshape(-1)
print(predict_list)
truth_list = numpy.array(truth_list)
predict_list1 = (predict_list[non_zeros] >0)
truth_list1 = (truth_list[non_zeros] >0)
print(predict_list1)
print(truth_list1)