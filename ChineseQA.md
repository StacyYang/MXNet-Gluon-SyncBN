### 中文常见问答
- 模型是否可以 Hybridize？

  训练不行，测试可以。
  MXNet相对于其他平台的显著优势就是提供两套API接口，Symbol API提供静态图速度快，NDArray／Gluon API是impreative执行，
  接口简单好用，而且可以通过hybridize加速训练，这样无缝连接了两套接口。
  目前跨卡BN只提供Gluon接口，在训练时候不能hbridize，
  不过在训练完成之后，BatchNorm在inference的时候不需要跨卡，可以转成普通BN来hybridize。

- 能否使用Symbol API？

  目前不行。我们后台增加的operators都是支持Symbol和NDArray两个接口的，所以在构建图的时候完成跨卡操作在理论上是可行的。
  因为笔者是从Gluon API之后开始学习MXNet的，所以目前没有提供Symbol的调用方法，欢迎大家贡献解决方案。

- 训练是否会变慢，能否分布式训练？

  变慢是相对的，目前不能分布式。
  相同迭代次数的情况下，训练时间会变长，因为同步的锅（overhead），但是我们往往可以加大learning rate，
  因为有了跨卡BN，梯度更加平缓了。
  目前已经有相关论文里面说到实现了128卡的分布式训练，证明是可行的，根据竟然这个跨卡BN的latency主要来自于同步，更接近于一个常数，
  所以在大运算量，大网络面前，相对的overhead变得很小。目前我们这个版本还不支持，但是后面可能会做。
  
欢迎大家到[论坛讨论区](https://discuss.gluon.ai/t/topic/1156)交流。
