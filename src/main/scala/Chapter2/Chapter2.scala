package Chapter2

/**
  * Created by Nyankosensei on 16/4/22.
  */
object Chapter2 extends App {

    /** Exercise2.1
      * 写一个递归函数,来获取第n个斐波那契数,前两个斐波那契数0和1,第n个数总是等于它前两个
      * 数的和--序列开始为0\1\1\2\3\5.应该定义为局部(local)尾递归函数.
      * def fib(n:Int):Int
      */

    //TODO 可以分析一下 斐波那契数的 各种算法 以及 尾递归转循环的算法
    def fib(n: Int): Int = {
        @annotation.tailrec
        def loop(n: Int, prev: Int, cur: Int): Int = {
            if (n == 0) prev
            else loop(n - 1, cur, prev + cur)
        }
        loop(n - 1, 0, 1) // TODO 这里貌似应该是 n-1 吧
    }

    /** Exercise2.2
      * 实现isSorted方法,检测Array[A]是否按照给定的比较函数排序:
      * def isSorted[A](as:Array[A],ordered:(A,A)=>Boolean):Boolean
      */
    def isSorted[A](as: Array[A], ordered: (A, A) => Boolean): Boolean = {

        def go(n: Int): Boolean = {
            if (n >= as.length - 1) true
            else if (!ordered(as(n), as(n + 1))) false //TODO 这里难道不应该是! ? 我在 这里加了个!
            else go(n + 1)
        }
        go(n = 0)
    }

    /** Exercise2.3
      * 我们来看一个柯里化(currying)的例子,把带有两个参数的函数f转换为只有一个参数的
      * 部分应用函数f.这里只有(要?)实现可编译通过
      * def curry[A,B,C](f:(A,B)=>C):A=>(B=>C)
      */
    def curry[A, B, C](f: (A, B) => C): A => (B => C) = a => b => f(a, b)

    /** Exercise2.4
      * 实现反柯里化(uncurry),与柯里化正相反.注意,因为右箭头=>是右结合的,
      * A=>(B=>C)可以写成A=>B=>C
      * def uncurry[A,B,C](f:A=>B=>C):(A,B)=>C
      */

    def uncurry[A, B, C](f: A => B => C): (A, B) => C = (a, b) => f(a)(b)

    /** Exercise2.4
      * 实现一个高阶函数,可以组合两个函数为一个函数
      * def compose[A,B,C](f:B=>C,g:A=>B):A=>C
      */

    def compose[A, B, C](f: B => C, g: A => B): A => C = a => f(g(a))
}

