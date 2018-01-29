import pandas as pd
one = [0,1,0,1]
two = ['one','two']
three = pd.Categorical.from_codes(one,two)
test = [[1,2,3],[2,3,4]]
pd_test = pd.DataFrame(test)
print(pd_test)
print(len(pd_test.iloc[0]))
