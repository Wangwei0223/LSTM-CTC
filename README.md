# LSTM-CTC

This project is based on Tensorflow, showing how to use basic CNN and RNN to process images as inputs to the CTC layer. 

By using the CTC layer, we are able to process label sequence without segregation for each character.

As for CNN layer, we defined conv layers and max_pool layer also you can apply BatchNormalization(You may get a better result).

You could see the details of CNN in CRNN.CTC.py
---------------------------------------------------------------------------
For RNN layer, you can choose Basic LSTM or Bidirection LSTM

You can add CTC layer to your model by applying CTC_loss in Tensorflow

loss = tf.nn.ctc_loss(targets, logits, seq_len)

You could see the details in train_CRNN.py
---------------------------------------------------------------------------
