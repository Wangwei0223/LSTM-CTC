# LSTM-CTC
This project is based on Tensorflow, showing how to use basic CNN and RNN to process images as inputs to the CTC layer. 
By using the CTC layer, we are able to process label sequence without segregation for each character.
As for CNN layer, we defined conv layers and max_pool layer also you can apply BatchNormalization(You may get a better result).

# conv1 = k:3*3, s:1, p:1 window 2*2
    W_conv1 = weight_variable([3, 3, 1, 64])
    b_conv1 = bias_variable([64])

    h_conv1 = tf.nn.relu(con2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

# conv2 = k:3*3, s:1, p:1 window 2*2
    W_conv2 = weight_variable([3, 3, 64, 128])
    b_conv2 = weight_variable([128])

    h_conv2 = tf.nn.relu(con2d(h_pool1, W_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

# conv3 = k:3*3, s:1, p:1
    W_conv3 = weight_variable([3, 3, 128, 256])
    b_conv3 = weight_variable([256])

    h_conv3 = tf.nn.relu(con2d(h_pool2, W_conv3)+b_conv3)

# conv4 = k:3*3, s:1, p:1 window 1*2
    W_conv4 = weight_variable([3, 3, 256, 256])
    b_conv4 = weight_variable([256])

    h_conv4 = tf.nn.relu(con2d(h_conv3, W_conv4)+b_conv4)
    h_pool3 = max_pool_1x2(h_conv4)

# conv5 = k:3*3, s:1, p:1
    W_conv5 = weight_variable([3, 3, 256, 512])
    b_conv5 = weight_variable([512])

    h_conv5 = tf.nn.relu(con2d(h_pool3, W_conv5)+b_conv5)

# BatchNormalization (Optional)
    # h_conv5 = tf.nn.batch_normalization(h_conv5)

# conv6 = k:3*3, s:1, p:1
    W_conv6 = weight_variable([3, 3, 512, 512])
    b_conv6 = weight_variable([512])

    h_conv6 = tf.nn.relu(con2d(h_conv5, W_conv6)+b_conv6)

# BatchNormalization
#    h_conv6 = tf.nn.batch_normalization(h_conv6)

    h_pool4 = max_pool_1x2(h_conv6)

# conv7 = k: 3*3, s:1, p:0
    W_conv7 = weight_variable([2, 2, 512, 512])
    b_conv7 = weight_variable([512])

    h_conv7 = tf.nn.relu(con2d_valid(h_pool4, W_conv7)+b_conv7)

# h_conv7 [128, 1, 6, 512]
    return tf.squeeze(h_conv7)
---------------------------------------------------------------------------
For RNN layer, you can choose Basic LSTM or Bidirection LSTM

You can add CTC layer to your model by applying CTC_loss in Tensorflow

loss = tf.nn.ctc_loss(targets, logits, seq_len)

You could see the details in train_CRNN.py
