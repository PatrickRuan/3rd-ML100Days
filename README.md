# 3rd-ML100Days

Day 01 (可以忽略，只是練習 產生資料 np.linspace(0,100,101)  np.random.randn(101)
    

    y = w*(x+n1a*n1)+b
    yp=w*x+b

    plt.plot(x,y, 'y.',label="data")
    plt.plot(x,yp,'b-',label='predict')
    plt.legend(loc=2)One Hot Encoding for dataframe 
     


Day 06

    One Hot Encoding for dataframe 
      df = pd.get_dummies(df)
      
    One Hot Encoding for array
      from keras.utils import to_categorical
      data_y = to_categorical(data_y)
