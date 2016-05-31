package Chapter5

import scala.annotation.tailrec

/**
  * Created by Nyankosensei on 16/5/19.
  */

import Stream._

object Chapter5 extends App {

    /** Exercise5.1
      * 写一个可将Stream转换成List的函数,他会被强制求值,可以在REPL下看到值的
      * 内容。可以转换成标准库中的常规List类型,可以把这个函数以及其他操作Stream
      * 的函数放到Stream特质内部。
      * def toList: List[A]
      * ---------------------------------------------------------------
      * trait Stream 中的toListRecursive、toList、toListFast函数
      */

    /** Exercise5.2
      * 写一个函数take(n)返回Stream中的前n个元素;写一个函数drop(n)返回
      * Stream中第n个元素之后的所有元素。
      * ---------------------------------------------------------------
      * trait Stream 中的take/drop函数
      */


    /** Exercise5.3
      * 写一个函数takeWhile返回Stream中从其实元素连续满足给定断言的所有元素。
      * def takeWhile(p: A => Boolean): Stream[A]
      * ---------------------------------------------------------------
      * trait Stream 中的take、drop函数
      */

    /** Exercise5.4
      * 实现一个forAll函数,检查Stream中所有元素是否与给定的断言匹配。遇到不匹
      * 配的值应该立即终止遍历。
      * def forAll(f: A => Boolean): Boolean
      * ---------------------------------------------------------------
      * trait Stream 中的forAll函数
      */

    /** Exercise5.5
      * 使用foldRight实现takeWhile
      * ---------------------------------------------------------------
      * trait Stream 中的takeWhile_1函数
      */

    /** Exercise5.6 <难>
      * 使用foldRight实现headOption。
      * ---------------------------------------------------------------
      * trait Stream 中的headOption函数
      */

    /** Exercise5.7
      * 用foldRight实现map、filter、append和flatMap,append方法参数应该是
      * 非严格求值的。
      * ---------------------------------------------------------------
      * trait Stream 中的map、filter、append、flatMap函数
      */

    /** Exercise5.8
      * 对ones稍微泛化一下,定义一个constant函数,根据给定值返回一个无限流
      * def constant[A](a: A): Stream[A]
      * ---------------------------------------------------------------
      * object Stream 中的constant函数
      */

    /** Exercise5.9
      * 写一个函数生成一个整数无限流,从n开始,然后n+1/n+2,等等。
      * def from(n: Int): Stream[Int]
      * ---------------------------------------------------------------
      * object Stream 中的from函数
      */

    /** Exercise5.10
      * 写一个fibs函数生成斐波那契数列的无限流:0,1,1,2,3,5,8
      * ---------------------------------------------------------------
      * object Stream 中的val fib
      */

    /** Exercise5.11
      * 写一个更加通用的构造流的函数unfold。它接收一个初始状态,以及一个在生成的
      * Stream中用于产生下一状态和下一个值的函数。
      * def unfold[A, S](z: S)(f: S => Option[(A, S)]): Stream[A]
      * ---------------------------------------------------------------
      * object Stream 中的unfold函数
      */

    /** Exercise5.12
      * 根据unfold函数来写fibs/from/constant和ones。
      * ----------------------------------------------------------------------------------
      * object Stream 中的fibsViaUnfold/formViaUnfold/constantViaUnfold/onesViaUnfold函数
      *
      */

    /** Exercise5.13
      * 使用unfold实现map/take/takeWhile/zipWith以及zipAll。
      * zilAll函数应该遍历只要stream还有更多元素--它使用Option
      * 表示Stream是否已经彻底遍历完了。
      * def zipAll[B](s2: Stream[B]): Stream[(Option[A], Option[B])]
      * ---------------------------------------------------------------
      * trait Stream 中的map/take/takeWhile/zipWith/zipAll函数
      *
      */

    /** Exercise5.14 <难>
      * 使用已写过的函数实现startsWith函数。它检查一个Stream是否是另一个
      * Stream的前缀,比如Stream(1,2,3) startsWith Stream(1,2)返回true。
      * def startsWith[A](s: Stream[A]): Boolean
      * ---------------------------------------------------------------
      * trait Stream 中的startsWith函数
      */

    /** Exercise5.15
      * 使用unfold实现tails函数,对一个给定的Stream,tails返回这个Stream
      * 输入序列的所有后缀(包含原始Stream),比如给定的Stream(1,2,3)返回
      * Stream(Stream(1,2,3),Stream(2,3),Stream(3),Stream())。
      * ---------------------------------------------------------------
      * trait Stream 中的tails函数
      */

    /** Exercise5.16 <难>
      * 把tails泛化为scanRight函数,类似foldRight返回一个中间结果的
      * Stream,例如:
      * scala> Stream(1,2,3).scanRight(0)(_+_).toList
      * res0> List[Int] = List(6,5,3,0)
      * ---------------------------------------------------------------
      * trait Stream 中的scanRight函数
      *
      */
}

trait Stream[+A] {
    def toListRecursive: List[A] = this match {
        case Cons(h, t) => h() :: t().toListRecursive
        case _ => List()
    }

    def toList: List[A] = {
        @tailrec
        def go(s: Stream[A], acc: List[A]): List[A] = s match {
            case Cons(h, t) => go(t(), h() :: acc)
            case _ => acc
        }
        go(this, List()).reverse
    }

    def toListFast: List[A] = {
        val buf = new collection.mutable.ListBuffer[A]
        def go(s: Stream[A]): List[A] = s match {
            case Cons(h, t) =>
                buf += h()
                go(t())
            case _ => buf.toList
        }
        go(this)
    }

    def take(n: Int): Stream[A] = this match {
        case Cons(h, t) if n > 1 => cons(h(), t().take(n - 1))
        case Cons(h, _) if n == 1 => cons(h(), empty) //这里可直接调用是因为import了 Stream半生对象
        case _ => empty
    }

    @tailrec
    final def drop(n: Int): Stream[A] = this match {
        case Cons(h, t) if n > 0 => t().drop(n - 1)
        case _ => this
    }

    def takeWhile(f: A => Boolean): Stream[A] = this match {
        case Cons(h, t) if f(h()) => cons(h(), t().takeWhile(f))
        case _ => empty
    }

    def foldRight[B](z: => B)(f: (A, => B) => B): B =
        this match {
            case Cons(h, t) => f(h(), t().foldRight(z)(f))
            case _ => z
        }

    def exists(p: A => Boolean): Boolean =
        foldRight(false)((a, b) => p(a) || b)

    def forAll(f: A => Boolean): Boolean =
        foldRight(true)((a, b) => f(a) && b)


    def takeWhile_1(f: A => Boolean): Stream[A] =
        foldRight(empty[A])((h, t) => if (f(h)) cons(h, t) else empty)

    def headOption: Option[A] =
        foldRight(None: Option[A])((h, _) => Some(h))


    def map[B](f: A => B): Stream[B] =
        foldRight(empty[B])((h, t) => cons(f(h), t))

    def filter(f: A => Boolean): Stream[A] =
        foldRight(empty[A])((h, t) => if (f(h)) cons(h, t) else t)

    def append[B >: A](s: => Stream[B]): Stream[B] =
        foldRight(s)((h, t) => cons(h, t))

    def flatMap[B](f: A => Stream[B]): Stream[B] =
        foldRight(empty[B])((h, t) => f(h) append t)


    def mapViaUnfold[B](f: A => B): Stream[B] =
        unfold(this) {
            case Cons(h, t) => Some(f(h()), t())
            case _ => None
        }

    def takeViaUnfold(n: Int): Stream[A] =
        unfold((this, n)) {
            case (Cons(h, t), 1) => Some((h(), (empty, 0)))
            case (Cons(h, t), nn) if nn > 1 => Some((h(), (t(), n - 1)))
            case _ => None
        }

    def takeWhileViaUnfold(f: A => Boolean): Stream[A] =
        unfold(this) {
            case Cons(h, t) if h() => Some(h(), t())
            case _ => None
        }

    def zipWith[B, C](s2: Stream[B])(f: (A, B) => C): Stream[C] =
        unfold((this, s2)) {
            case (Cons(h1, t1), Cons(h2, t2)) =>
                Some((f(h1(), h2()), (t1(), t2())))
            case _ => None
        }

    def zip[B](s2: Stream[B]): Stream[(A, B)] =
        zipWith(s2)((_, _))


    def zipWithAll[B, C](s2: Stream[B])(f: (Option[A], Option[B]) => C): Stream[C] =
        unfold((this, s2)) {
            case (Empty, Empty) => None
            case (Cons(h, t), Empty) => Some(f(Some(h()), Option.empty[B]) ->(t(), empty[B]))
            case (Empty, Cons(h, t)) => Some(f(Option.empty[A], Some(h())) -> (empty[A] -> t()))
            case (Cons(h1, t1), Cons(h2, t2)) => Some(f(Some(h1()), Some(h2())) -> (t1() -> t2()))
        } // 写出这个的也是牛逼完全不想看了

    def zipAll[B](s2: Stream[B]): Stream[(Option[A], Option[B])] =
        zipWithAll(s2)((_, _))


    def startsWith[A](s: Stream[A]): Boolean =
        zipAll(s).takeWhile(_._2.isDefined) forAll {
            case (h, h2) => h == h2
        }

    def tails: Stream[Stream[A]] =
        unfold(this) {
            case Empty => None
            case s => Some((s, s drop 1))
        } append Stream(empty)

    def hasSubSequence[A](s: Stream[A]): Boolean =
        tails exists (_ startsWith s)

    def scanRight[B](z: B)(f: (A, => B) => B): Stream[B] =
        foldRight((z, Stream(z)))((a, p0) => {
            lazy val p1 = p0
            val b2 = f(a, p1._1)
            (b2, cons(b2, p1._2))
        })._2

    @tailrec
    final def find(f: A => Boolean): Option[A] = this match {
        case Empty => None
        case Cons(h, t) => if (f(h())) Some(h()) else t().find(f)
    }
}

case object Empty extends Stream[Nothing]

case class Cons[+A](h: () => A, t: () => Stream[A]) extends Stream[A]

object Stream {

    def cons[A](hd: => A, tl: => Stream[A]): Stream[A] = {
        lazy val head = hd
        lazy val tail = tl
        Cons(() => head, () => tail)
    }

    def empty[A]: Stream[A] = Empty

    def apply[A](as: A*): Stream[A] = as match {
        case Nil => empty
        case (head, tail) => cons(head, apply(tail: _*))
    }

    val ones: Stream[Int] = Stream.cons(1, ones)

    def constant[A](a: A): Stream[A] = {
        lazy val tail: Stream[A] = Cons(() => a, () => tail)
        tail
    }

    def from(n: Int): Stream[Int] = cons(n, from(n + 1))

    val fib = {
        def go(f0: Int, f1: Int): Stream[Int] =
            cons(f0, go(f1, f0 + f1))
        go(0, 1)
    }

    def unfold[A, S](z: S)(f: S => Option[(A, S)]): Stream[A] =
        f(z) match {
            case Some((a, s)) => cons(a, unfold(s)(f))
            case None => empty
        }

    def unfoldViaFold[A, S](z: S)(f: S => Option[(A, S)]): Stream[A] =
        f(z).fold(empty[A])((p: (A, S)) => cons(p._1, unfold(p._2)(f)))


    def unfoldViaMap[A, S](z: S)(f: S => Option[(A, S)]): Stream[A] =
        f(z).map((p: (A, S)) => cons(p._1, unfold(p._2)(f))).getOrElse(empty[A])


    val fibsViaUnfold =
        unfold((0, 1)) { case (f0, f1) => Some((f0, (f1, f0 + f1))) }

    def formViaUnfold(n: Int) =
        unfold(n)(n => Some((n, n + 1)))

    def constantViaUnfold[A](a: A) =
        unfold(a)(_ => Some((a, a)))

    val onesViaUnfold = unfold(1)(_ => Some(1, 1))
}