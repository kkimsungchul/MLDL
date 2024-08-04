#import tensorflow as tf
import tensorflow.compat.v1 as tf

# v1로 돌릴때 오류가 발생함
# 오류메시지 : RuntimeError: `loss` passed to Optimizer.compute_gradients should be a function when eager execution is enabled.
# 참고링크1 : https://stackoverflow.com/questions/57858219/loss-passed-to-optimizer-compute-gradients-should-be-a-function-when-eager-exe
# 참고링크2 : https://xengom.tistory.com/5
# 해결방법 :
# import 를 변경 -> import tensorflow.compat.v1 as tf
# v2 비활성화 -> tf.disable_v2_behavior()

# TensorFlow
# 1. 그래프 빌드
# 2. 세션을 통해서 그래프 실행
# 3. 실행 결과로 그래프 업데이트
# placeholder를 사용하는 가장 큰 이유는 만들어진 모델에 대해서 값을 따로 넘겨줄수 있다.

# TensorFlow 로 그래프를 생성
# 만들어진 모들을 사용하기 위해 feed_dict방법을 이용해 x,y 데이터를 줌
# 결과적으로 w,b 데이터 업데이트

# H(x) = Wx +b

tf.disable_v2_behavior()


# x_train = [1, 2, 3]
# y_train = [1, 2, 3]
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#hypothesis = x_train * w + b
hypothesis = X * W + b
#cost = tf.reduce_mean(tf.square(hypothesis - y_train))
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    # sess.run(train)
    # if step % 20 == 0:
    #     print(step, sess.run(cost), sess.run(w), sess.run(b))

    cost_val, W_val, b_val , _ = sess.run([cost, W, b, train],
        feed_dict = {X:[1, 2, 3, 4, 5],
                     Y : [2.1, 3.1, 4.1, 5.1, 6.1]})
    if step % 20 == 0:
        print(step, cost_val , W_val , b_val)

print('------------------------')
print(sess.run(hypothesis , feed_dict = {X: [5]}))
print(sess.run(hypothesis , feed_dict = {X: [2.5]}))
print(sess.run(hypothesis , feed_dict = {X: [1.5 , 3.5]}))
