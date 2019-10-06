# 3rd-ML100Days

# 順一下， day 1 ~ day 14 觀察 dataframe, 然後思考觀察離群值
    
    1.) Day 4 我們拿到 DataFrame, 先看好 shape, describe(), info(), value_counts(), dtype, iloc slicing 了解我們的資料有多少，是什麼
    2.) 經由 df.dtypes.value_counts() 將資料分成 數值，物件 常用 df_num = df[num_list], 
        進行 one hot encoding
        進行 label encoding (如果只有兩個 outcome)
    3.) Outlier 對於 numeric data, 我們移除二值(通常是 0, 1), 進行直方圖或相形圖觀察，也作 cdf 的觀察。要花時間看，用心想，資料探勘，不是學程式而是要用心看資料，看程式處理好的格式的資料。
    4.) Outlier 可以用 percentile, 將max 修改為 q99, 或 q50 NA 填補/ 平均數 (mean) /中位數 (median, or Q50)/ 最大/最小值 (max/min, Q100, Q0)/ 分位數 (quantile)
    5.) 缺失值補值 常用 isnull(), fillna(), 簡單方式是 fillna(0), fillna(-1), fillna(df.mean()), 
    
    拿到資料，觀察行列多寡外型，統計量，是數值還是string, 作 LE, OHE, 然後看 outlier 乖乖看直方圖。cdf，然後將怪怪值補 -1, mean(), median, mode(), q99, q50, or max, min, or NA... 補缺失值，先告一段落，我們來作 
    ***我們也會拿 不同結果 如 target=1, target =2 作圖比較，比如必較 target=1, target =2 的 cdf 看看分布有沒有不同... 
    6.) correlation
    7.) 差不多了，可能可以試試看 簡化模型，將連續值化成離散階層，用 pd.cut(df[col], bins)
    
*** D1~ D6，
    
    我們拿到資料，先看 shape, describe (mean, max, min...), dtype, and go ohe 

*** D7 ~ D10，

    我們開始看 觀察 資料欄位的類型求數值欄的平均值等，物件欄的數量等。
    用直方圖觀察異常、用 cdf 留白觀察異常值，也嘗試了 clip data or 去除異常值。
    用 cross_val_score(...,cv=5).mean() 比較。

*** D11 ~ D13  

    我們的缺失值可以用 -1, mean()，直接刪去，.clip(300,2000)，
    Outlier 也可以用 percentile, 將max 修改為 q99, 
    或 q50 NA 填補/ 平均數 (mean) /中位數 (median, or Q50)/ 最大/最小值 (max/min, Q100, Q0)/ 分位數 (quantile)

*** D14 ~ D16，
    
    做兩兩資料欄的相關係數與散射圖，與後頭會介紹的 Heatmap 相關不相同。
    然後發揚光大 hist() 這一派，加入 sns.kdeplot() 可以將各種 labeloutcome 放在一張圖上去練習解釋。

**** Day 17 ~ Day 20 (EDA finished), 

    17, 18 在練習離散化資料，跟 quantization 類似吧！　
    19 subplot 繪圖技巧， 
    20 heatmap and pairgrid. 


    
# 程式學習點
    
    Day 5 的三種讀圖檔，一種存圖檔
    Day 7 reset_index() 可以長成 dataframe、groupby().aggregate('count')
    Day 9 cdf 
        
        # 再把只有 2 值 (通常是 0,1) 的欄位去掉
        numeric_columns = list(app_train[numeric_columns].columns[list(app_train[numeric_columns].apply(lambda x:len(x.unique())!=2 ))])
        cdf = app_train['AMT_INCOME_TOTAL'].value_counts().sort_index().cumsum()
        plt.plot(list(cdf.index), cdf/cdf.max())   # 改變 y 軸的 Scale, 讓我們可以正常檢視 ECDF
    Day 10   cross_val_score(estimator, train_X, train_Y, cv=5).mean()
        # 將 1stFlrSF 限制在你覺得適合的範圍內, 捨棄離群值
        keep_indexs = (df['1stFlrSF']> 800) & (df['1stFlrSF']< 2500)
        df = df[keep_indexs]
    Day 11 q100 = [np.percentile(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'],q= i) for i in range(101)]
    Day 19
        plt.figure(figsize=(8,6))
        for i in range(len(year_group_sorted)):
        sns.distplot(age_data.loc[(age_data['YEARS_BINNED'] == year_group_sorted[i]) & 
                              (age_data['TARGET'] == 0), 'YEARS_BIRTH'], label = str(year_group_sorted[i]))
    
        sns.distplot(age_data.loc[(age_data['YEARS_BINNED'] == year_group_sorted[i]) & 
                              (age_data['TARGET'] == 1), 'YEARS_BIRTH'], label = str(year_group_sorted[i]))
        plt.title('KDE with Age groups')

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

    One Hot Encoding for dataframe   # df is a DataFrame
      df = pd.get_dummies(df)
      
    One Hot Encoding for array      # data_y is a numpy object.  
      from keras.utils import to_categorical
      data_y = to_categorical(data_y)


Day 07 欄位類別， 
    
    # 關於 reset_index() 很精彩
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
    

    
Day 11     處理 outliers/ 新增欄位註記/ outliers 或 NA 填補/ 平均數 (mean) /中位數 (median, or Q50)/ 最大/最小值 (max/min, Q100, Q0)/ 分位數 (quantile)
    
    q100 = [np.percentile(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'],q= i) for i in range(101)]
    拆解: 
    我們需要一個 把 app_train 中間 isnull 標記為 fales, "isntnull" 標記為 true 的 serie, 所以用 ~app_train['AMT_ANNUITY'].isnull()
    所以把 app_train[~app_train['AMT_ANNUITY'].isnull()] 就會得到 app_train 中 只有 AMT_ANNUITY 非空值的行(筆)資料的 dataframe
    上面的 dataframe 再加上 ['AMT_ANNUITY'] 就會是 app_train['AMT_ANNUITY'] 非空值的 列(欄)資料
        = app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY']
    要作 percentile 要有兩個參數 (serie data, q=多少階)
    [np.percentile(~, q=i) for i in range(101)]  <== 這是一個 list?
    
    app_train[app_train['AMT_ANNUITY'] == app_train['AMT_ANNUITY'].max()] = np.percentile(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'], q = 99)
    
    q_50 = q_all[50]
    app_train.loc[app_train['AMT_ANNUITY'].isnull(),'AMT_ANNUITY'] = q_50
    
    def normalize_value(x):
        Max = x.max()
        Min = x.min()
        x = 2* ((x-Min)/(Max-Min))-1
        return x
    app_train['AMT_ANNUITY_NORMALIZED'] = normalize_value(app_train['AMT_ANNUITY'])
 
    mode_goods_price = list(app_train['AMT_GOODS_PRICE'].value_counts().index)
    app_train.loc[app_train['AMT_GOODS_PRICE'].isnull(), 'AMT_GOODS_PRICE'] = mode_goods_price[0]
    
 
    
    
Day 12 缺失值與標準化

        填補統計值
            •填補平均值(Mean) : 數值型欄欄位，偏態不明顯    fillna(df.mean())
            •填補中位數(Median) : 數值型欄欄位，偏態很明顯   
            •填補眾數(Mode) : 類別型欄欄位
        填補指定值 - 需對欄欄位領域知識已有了了解
            •補 0 : 空缺原本就有 0 的含意，如前⾴頁的房間數   fillna())
            •補不可能出現的數值 : 類別型欄欄位，但不適合⽤用眾數時
        填補預測值 - 速度較慢但精確，從其他資料欄欄位學得填補知識
            •若若填補範圍廣，且是重要特徵欄欄位時可⽤用本⽅方式
            •本⽅方式須提防overfitting : 可能退化成為其他特徵的組合
        
        其他處理 MinMaxScaler(), StandardScaler()
        stds.fit_transform(df)

Day 13 Dataframe operation
    
    concat, mearge, melt
    sub_df = app_train.loc[app_train['AMT_INCOME_TOTAL'] > app_train['AMT_INCOME_TOTAL'].mean(), ['SK_ID_CURR', 'TARGET']]
    # 應該要熟悉上面的寫法，loc 中第一項是在作 true or false 的行，資料筆數的篩選，底下兩行都是類似
    # app_train.loc[app_train['AMT_ANNUITY'].isnull(),'AMT_ANNUITY'] = q_50
    # [np.percentile(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'],q= i) for i in range(101)]
    # 取前 10000 筆作範例: 分別將 AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY 除以根據 NAME_CONTRACT_TYPE 分組後的平均數，
    app_train.loc[0:10000, ['NAME_CONTRACT_TYPE', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']].groupby(['NAME_CONTRACT_TYPE']).apply(lambda x: x / x.mean())
    (lambda x:(x-np.mean(x))/np.std(x))  # z-transform


Day 14 相關係數
    
    np.corrcoef(x,y), y = x+np.random.normal(0,10,1000), x = np.random.randint(0,50,1000)
    
Day 15, 

    相關兩資料(兩欄)的點圖 plt.plot(sub_df['DAYS_EMPLOYED'] / (-365), sub_df['AMT_INCOME_TOTAL'], '.')， 搭配 np.corrcoef(x,y)
    
Day 16, 
    
    利用 sns.kdeplot(x1) 作出類似 plt.hist() 的分布，也鼓勵用 kdeplot(x1, target='1') pk target ='0', 因為單純一條線，所以同圖看很清楚。


Day 17 Day 18  連續值離散化 參數下降，可以... 簡單化模型?
    
    ages["equal_width_age"] = pd.cut(ages["age"], 4)
    ages["equal_width_age"] = pd.qcut(ages["age"], 4)
    bins = pd.IntervalIndex.from_tuples([(0, 10), (10, 20), (20, 30), (30, 50), (50, 100)])
    ages['customized_age_grp']=pd.cut(ages['age'], bins)
    
Day 19 subplot, 

    #起手式
    plt.figure(figsize=(8,8))
    plt.subplot(321)
    # 比較有趣的是，將資料分組，畫在同一張圖上，一張圖具有 subplot 的效果:繪製分群後的 10 條 KDE 曲線
    
Day 20 heatmap:  

    sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
    grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey=False,
                    hue = 'TARGET', vars = [x for x in list(plot_data.columns) if x != 'TARGET'])
    # 上半部為 scatter
    grid.map_upper(plt.scatter, alpha = 0.2)
    # 對角線畫 histogram
    grid.map_diag(sns.kdeplot)
    # 下半部放 density plot
    grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r)
    

is.null() 專區

    # 檢查欄位缺值數量 (去掉.head()可以顯示全部)
    # 這樣的 isnull() 是對欄位作檢查，也就是 
    df.isnull().sum().sort_values(ascending=False).head()
    
    app_train['AMT_ANNUITY'].isnull(), 沒有意外，這個只會對單一欄位啊!  這是在作 np.percentile 時用到，馬上練習一遍
        [np.percentile(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'], q =i) for i in range(101)] 
        
    app_train.loc[app_train['AMT_ANNUITY'].isnull(),'AMT_ANNUITY'] = q_50  這個是用 單欄布林表挑選遺失值 資料 行，在限定在 該欄位， loc 用法。
        
        
    
