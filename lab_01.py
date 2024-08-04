import tensorflow as tf
# 버전 문제 발생시 import 를 아래와 같이 변경
# import tensorflow.compat.v1 as tf
# github에 tf2.x 버전 이상의 코드도 있음
# https://github.com/hunkim/DeepLearningZeroToAll/tree/master/tf2



# 그래프를 정의함
# placeholder를 만들수 있음
# 세션을 통해서 실행할 때 값을 넘겨줌
# 그래프가 실행되면서 값을 업데이트 하거나, 값을 리턴해줌

# 그래프를 설계 하고
# 그래프를 실행하고
# 그 결과로 그래프의 결과가 변화하거나 하는는걸 알수 있음


# youtube 댓글 2
#2.0버전에서 Session 모듈을 더이상 지원하지 않네요. 간단하게 print를 실습해보고 싶으시면
#hello = tf.constant("Hello, TensorFlow!")
#tf.print(hello)
#tf.print(tf.add([1,3],[4,6]))

print("-------------------------")
node1 = tf.constant(3.0)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
print(node3)

print("-------------------------")

#a = tf.placeholder(tf.float32)
#b = tf.placeholder(tf.float32)
#c = tf.add(a, b)
#print(c)

# youtube 댓글 1
# tensorflow 2.0 버전 부터는 placeholder를 사용하지않고,  @tf.function 으로 함수를 정의해서 사용하는 것 같습니다
@tf.function
def adder(a,b):
   return a+b
a = tf.constant([1,3])
b = tf.constant([2,4])
print(adder(a,b))

