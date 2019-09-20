# 3rd-ML100Days

Day 01 (可以忽略，只是練習 產生資料 np.linspace(0,100,101)  np.random.randn(101)
    

    y = w*(x+n1a*n1)+b
    yp=w*x+b

    plt.plot(x,y, 'y.',label="data")
    plt.plot(x,yp,'b-',label='predict')
    plt.legend(loc=2)One Hot Encoding for dataframe 
   
Day 02 (可以忽略，認識 Machine Learning)

Day 03 (可以忽略，了解各公司是怎麼應用機器學習在實際的專案上)

Day 04 觀察資料觀察 DataFrame，Training Data, 先看 shape, desribe(), dtypes.value_counts(), app_train.iloc[300:310,1:5]
    
    df_train = pd.read_csv(data_path + 'house_train.csv.gz')
    train_Y = np.log1p(df_train['SalePrice'])
    df = df_train.drop(['Id', 'SalePrice'] , axis=1)
 
Day 5 產生 DataFrame 的方法
    
    Method 1, 
    data = {'國家': range(5) ,
    '人口':np.random.randint(prmin, prmax, 5) }
    data = pd.DataFrame(data)
    Method 2, 
    # zip mothed
    col1=[]
    col2=[]
    for i in range(5):
        country = 'country'+str(i)
        col1.append(country)
        col2.append(np.random.randint(prmin,prmax,1))
    data = [col1, col2]
    df = list(zip(label,data))
    label = ['country','population']
    df = pd.DataFrame(dict(df))
    
    還是要介紹一下類似爬蟲的文句切割  
    import requests
    response = requests.get(target_url)
    data = response.text
    split_tag = '\n'
    data = data.split(split_tag)
    
    col1 = []
    col2 = []
    data_row =[]
    count=0
    data=data[:len(data)-1]

    split_tag ='\t'
    for i in data:   
        data_row = i.split(split_tag)
        col1.append(data_row[0])
        col2.append(data_row[1])
    arrange_data={'Member':col1,'Photo':col2}
    df = pd.DataFrame(arrange_data)
    
    讀圖檔方法一
    from PIL import Image
    from io import BytesIO
    # 請用 df.loc[...] 得到第一筆資料的連結
    first_link = df.loc[0,'Photo']
    response = requests.get(first_link)
    img = Image.open(BytesIO(response.content))
    # Convert img to numpy array
    plt.imshow(img)
    plt.show()
    讀圖檔方法一
    import skimage.io as skio
    img1 = skio.imread('data/examples/example.jpg')
    plt.imshow(img1)
    plt.show()
    讀圖檔方法二
    from PIL import Image
    img2 = Image.open('data/examples/example.jpg') # 這時候還是 PIL object
    img2 = np.array(img2)
    plt.imshow(img2)
    plt.show()
    讀圖檔方法三
    import cv2
    img3 = cv2.imread('data/examples/example.jpg')
    plt.imshow(img3)
    plt.show()

    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    plt.imshow(img3)
    存圖檔方法
    import scipy.io as sio
    sio.savemat(file_name='data/examples/example.mat', mdict={'img': img1})
    mat_arr = sio.loadmat('data/examples/example.mat')
    mat_arr = mat_arr['img']
    plt.imshow(mat_arr)
    plt.show()



Day 06 One Hot Encoding,

    One Hot Encoding for dataframe 
      df = pd.get_dummies(df)
      
    One Hot Encoding for array
      from keras.utils import to_categorical
      data_y = to_categorical(data_y)

Day 07 欄位類別， 
    
    df.dtypes.value_counts() 相當
    dtype_df = df.dtypes.reset_index()
    dtype_df.columns = ["Count", "Column Type"]
    dtype_df = dtype_df.groupby("Column Type").aggregate('count').reset_index()
    dtype_df

    df_test = df_test.drop(['PassengerId'] , axis=1)
    df = pd.concat([df_train,df_test])
    
    for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64':
        float_features.append(feature)
    df[int_features].mean()
    df[int_features].max()
    df[object_features].unique() failed
    df[object_features].nunique()
    
Day 08 [作業目標]  對資料做更多處理 : 顯示特定欄位的統計值與直方圖   
    
    app_train.AMT_ANNUITY.hist()
    
Day 09 異常偵測，

    # 再把只有 2 值 (通常是 0,1) 的欄位去掉
    numeric_columns = list(app_train[numeric_columns].columns[list(app_train[numeric_columns].apply(lambda x:len(x.unique())!=2 ))])
    
    app_train[col].hist()
    plt.xlabel(col) 
    製作 CDF
    cdf = app_train['AMT_INCOME_TOTAL'].value_counts().sort_index().cumsum()
    plt.plot(list(cdf.index), cdf/cdf.max())
    # 改變 y 軸的 Scale, 讓我們可以正常檢視 ECDF
    plt.plot(np.log(list(cdf.index)), cdf/cdf.max())
    # 小技巧
    plt.xlim([cdf.index.min(), cdf.index.max() * 1.05]) # 限制顯示圖片的範圍
    plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍
    
    注意：當 histogram 畫出上面這種圖 (只出現一條，但是 x 軸延伸很長導致右邊有一大片空白時，代表右邊有值但是數量稀少。這時可以考慮用 value_counts 去找到這些數值
    
    
    
    
Day 10 
    # 顯示 1stFlrSF 與目標值的散佈圖
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.regplot(x = df['1stFlrSF'][:train_num], y=train_Y)
    plt.show()

    # 做線性迴歸, 觀察分數
    train_X = MMEncoder.fit_transform(df)
    estimator = LinearRegression()
    cross_val_score(estimator, train_X, train_Y, cv=5).mean()
        df['1stFlrSF'] = df['1stFlrSF'].clip(300, 2000)
        sns.regplot(x = df['1stFlrSF'], y=train_Y)
        plt.show()

    # 做線性迴歸, 觀察分數
    train_X = MMEncoder.fit_transform(df)
    estimator = LinearRegression()
    cross_val_score(estimator, train_X, train_Y, cv=5).mean()
    # 將 1stFlrSF 限制在你覺得適合的範圍內, 調整離群值

    df['1stFlrSF'] = df['1stFlrSF'].clip(300, 2000)
    sns.regplot(x = df['1stFlrSF'], y=train_Y)
    plt.show()


    # 做線性迴歸, 觀察分數
    train_X = MMEncoder.fit_transform(df)
    estimator = LinearRegression()
    cross_val_score(estimator, train_X, train_Y, cv=5).mean()
    
    # 將 1stFlrSF 限制在你覺得適合的範圍內, 捨棄離群值
    keep_indexs = (df['1stFlrSF']> 800) & (df['1stFlrSF']< 2500)
    df = df[keep_indexs]
    train_Y = train_Y[keep_indexs]
    sns.regplot(x = df['GrLivArea'], y=train_Y)
    plt.show()
    # 做線性迴歸, 觀察分數
    train_X = MMEncoder.fit_transform(df)
    estimator = LinearRegression()
    cross_val_score(estimator, train_X, train_Y, cv=5).mean()
    
    
    
順一下， day 1 ~ day 10 觀察 dataframe, 然後思考觀察離群值
    
    1.) Day 4 我們拿到 DataFrame, 先看好 shape, describe(), info(), value_counts(), dtype, iloc slicing 了解我們的資料有多少，是什麼
    2.) 經由 df.dtypes.value_counts() 將資料分成 數值，物件 常用 df_num = df[num_list], 
        進行 one hot encoding
        進行 label encoding (如果只有兩個 outcome)
    3.) 對於 numeric data, 我們移除二值(通常是 0, 1), 進行直方圖或相形圖觀察，也作 cdf 的觀察。
    
