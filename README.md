# Digits-Table-data-recognition- 專題實作競賽「讓額溫表降燒－表格資料辨識系統建置」專題開源程式碼

你好！感謝你來到這裡，這裡是我們的專案系統的開源程式碼，所有的程式碼並非由我個人所完成，在網路上我也參考了許多如何執行一些像是怎麼取得表格框架的方法，也就是運用直線與橫線取得相交點的方法，以下也會列出我所參考的程式碼來源與參考了哪裡，希望我們的專案也能給你一份參考！那麼廢話不多說，以下就先來介紹我們的專案程式做了什麼，又參考來自哪裡。

## 程式碼執行流程

首先就是我們的系統做了些什麼，他們的順序又是怎麼樣的，因為我們的程式碼有許多的function定義，所以直接看的話很難看出來程式的執行順序是怎麼樣的，那麼這裡就先簡單的介紹一下我們的系統是怎麼運作的吧！

1.    將圖片處理成乾淨的文字
2.    進行表格切割
3.    取得表格切割圖，將個別文字切割
4.    將切割的文字調整成正方形
5.    提供給機器進行辨識
6.    將辨識結果儲存至python迭代數據
7.    將儲存的辨識結果轉換成CSV檔並輸出

接下來怎麼辦到的可以在程式碼裡面查看，很多地方都有註解。如果有不理解的地方或著程式碼的編寫建議，可以選擇私訊我，或者在這裡開設Issues也可以！
