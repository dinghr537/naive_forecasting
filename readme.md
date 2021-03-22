## First about the environment

I'm using python 3.6 here for this homework (well python 3.6.13 exactly)

other packages can be installed by using pip $\rightarrow$ `pip install -r requirements.txt` 

if TA wants to train the model (not just using the existing model I've trained before, please follow the line 153 of the app.py)

## About data and the model

本次使用的兩組data，分別來自 https://data.gov.tw/dataset/19995 和 https://data.gov.tw/dataset/25850 ，都是點選csv那裡，下載下來的。

本次實作，暫時僅使用了提供的data中的備轉容量進行訓練。中途有思考過要加入使用天氣data進行train，不過沒能找到合適的過去的data，只能找到之後的天氣預告訊息，於是放棄了加入天氣相關資訊。

在資料前處理階段，將2020.csv中的備轉容量提取出來，之後去處其中包括的2021.1的內容（因為會和之後2021的資料重複），再將2021.csv中的備轉容量提取出來，乘10進行單位統一，再將結果與2020的data拼接在一起，得到可用的備轉容量data。

之後用每30天和之後8天的data生成出一組data，用來train（因為作業的ddl是3.22，而當天只能拿到21號的data，因此除了要預測的23～29號，22號的data也是要預測的，因此要預測八天）

模型則是參考網路上的部分推薦，先用了兩個lstm層，之後用timedistributed層dense了一下，再flatten成一維的data，最後再接兩個一維的dense層，最終得到8個target的vector。最後訓練完之後，將最近一個月的data輸入進模型，即可得到作業需要的結果。關於模型的調整優化，由於目前知識儲備有限，未能對模型進行客製化調整，只是沿用網路上的部分教程給出的結構，之後應該會有能力再進行優化。

