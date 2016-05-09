package Chapter3

import scala.annotation.tailrec

/**
  * Created by Nyankosensei on 16/4/27.
  */
object Chapter3 extends App {

    /** Exercise3.1
      * 下面的匹配表达式结果是什么?
      * val x = List(1, 2, 3, 4, 5) match {
      * case Cons(x, Cons(2, Cons(4, _))) => x
      * case Nil => 42
      * case Cons(x, Cons(y, Cons(3, Cons(4, _)))) => x + y
      * case Cons(h, t) => h + sum(t)
      * case _ => 101
      * }
      */

    val r = List(1, 2, 3, 4, 5) match {
        case Cons(x, Cons(2, Cons(4, _))) => x
        case Nil => 42
        case Cons(x, Cons(y, Cons(3, Cons(4, _)))) => x + y
        //case Cons(h, t) => h + sum(t) TODO 这里的sum 不知道哪来的
        case _ => 101
    }

    /** Exercise3.2
      * 实现tail函数,删除一个List的第一个元素.注意这个函数的时间开销是常量级的.
      * 如果列表是Nil,在实现的时候会有什么不同的选择?我们在下一章再回顾这个问题.
      * ---------------------------------------------------------------
      * object List 中的tail函数
      */

    List.tail(List(1, 2, 3, 4, 5))

    /** Exercise3.3
      * 使用相同的思路,实现函数setHead用一个不同的值代替列表中的第一个元素.
      * -----------------------------------------------------------
      * object List 中的setHead函数
      */

    List.setHead(List(1, 2, 3, 4, 5), 9)

    /** Exercise3.4
      * 把tail泛化为drop函数,用于从列表中删除前n个元素.注意,这个函数的时间
      * 开销只需要与drop的元素个数成正比--不需要复制整个列表
      * def drop[A](l:List[A],n:Int):List[A]
      * -----------------------------------------------------------
      * object List 中的drop函数
      */

    List.drop(List(1, 2, 3, 4, 5), 3)

    /** Exercise3.5
      * 实现dropWhile函数,删除列表中前缀全部符合判定的元素
      * def dropWhile[A](l:List[A],f:A=>Boolean):List[A]
      * -----------------------------------------------------------
      * object List 中的dropWhile函数
      */

    List.dropWhile(List(1, 1, 1, 1, 3, 3, 3, 3, 4, 4, 5), (x: Int) => x == 1)

    /** Exercise3.6
      * 不是所有的实现都这么令人满意,实现一个init函数,返回一个列表,它包含原列表中除了最后一个元素之外的所有元素.
      * 比如,传入List(1,2,3,4)给init函数会返回List(1,2,3),为什么这个函数不能实现同tail一样的常量级开销?
      * TODO 什么是常量级开销
      * def init[A](l:List[A]):List[A]
      * -------------------------------------------------------------------------------------------
      * object List 中的init函数
      */

    List.init(List(1, 2, 3, 4, 5, 6))

    /** Exercise3.7
      * 在入参是0.0时用foldRight实现product是否可以立即停止递归并返回0.0?
      * 为什么可以或者为什么不可以?想想看如果你对一个大列表调用foldRight会有多少短路发生.
      * 这个问题有点深,我们在第五章再来回顾.
      */

    /** Exercise3.8
      * 当你对foldRight传入Nil和Cons时,看看会发生什么?例如:
      * foldRight(List(1,2,3,4),Nil:List[Int])(Cons(_,_)).
      * 说到foldRight和List数据结构之间的关系,你有什么想法?
      */

    List.foldRight(List(1, 2, 3, 4), Nil: List[Int])(Cons(_, _))

    /** Exercise3.9
      * 使用foldRight计算List的长度
      * def length[A](as:List[A]):Int
      * -----------------------------
      * object List 中的length函数
      */

    List.length(List(1, 2, 3, 4, 5))

    /** Exercise3.10
      * 我们实现的foldRight不是尾递归,如果List很大可能会发生StackOverFlowError
      * (我们称之为非栈安全的).说服自己接受这种情况,然后用尾递归方式写另一个通用的
      * 列递归函数foldLeft.签名如下
      * def foldLeft[A,B](as:List[A],z:B)(f:(B,A)=>B):B
      * -----------------------------------------------------------------
      * object List 中的foldLeft函数
      */


    /** Exercise3.11
      * 写一下sum/product函数,和一个用foldLeft计算列表长度的函数
      * -----------------------------------------------------------------
      * object List 中的sum3/product3/length2函数
      */

    /** Exercise3.12
      * 写一个对原列表元素颠倒顺序的函数(List(1,2,3)返回List(3,2,1)),
      * 看看是否可用一种折叠(fold)实现
      * -----------------------------------------------------------------
      * object List 中的reverse函数
      */

    List.reverse(List(1, 2, 3))

    /** Exercise3.13 <难>
      * 你能否根据foldRight来写一个foldLeft?有没有其他变通的方式?
      * 通过foldLeft来实现foldRight很有用,因为这会让我们以尾递归的方式实现.
      * 意味着不管列表有多大都不会产生栈溢出
      * -----------------------------------------------------------------
      * object List 中的foldRightViaFoldLeft/foldRightViaFoldLeft_1/foldLeftViaFoldRight函数
      */

    /** Exercise3.14
      * 根据foldLeft或foldRight实现append函数
      * ---------------------------------------
      * object List 中的appendViaFoldRight/appendViaFoldLeft函数 TODO 这里如果用foldLeft来写的话需要反转第一个list 那样的话与用foldLeft实现foldRight相同
      */

    /** Exercise3.15 <难>
      * 写一个函数将一组列表连接成一个单个列表.他的运行效率
      * 应该随所有列表的总长度线性增长.试着用我们已经定义过的函数.
      * ---------------------------------------------------
      * object List 中的concat函数
      */

    /** Exercise3.16
      * 写一个函数,用来转换一个整数列表,对每个元素加1
      * (记住这应该是一个纯函数,返回一个新列表)
      * ---------------------------------------------------
      * object List 中的add1函数
      */

    /** Exercise3.17
      * 写一个函数,将List[Double]中的每一个值转换为String,
      * 你可以用表达式d.toString将Double类型的值转换为String
      * ---------------------------------------------------
      * object List 中的doubleToString函数
      */

    /** Exercise3.18
      * 写一个泛化的map函数,对列表中的每个元素进行修改,并维持列表结构
      * 签名如下:
      * def map[A,B](as:List[A])(f:A=>B):List[B]
      * ---------------------------------------------------
      * object List 中的map/map_me/map_1/map_2函数
      */

    /** Exercise3.19
      * 写一个filter函数,从列表中删除所有不满足断言的元素,
      * 并用它删除一个List[Int]中的所有奇数.
      * def filter[A](as:List[A])(f:A=>Boolean):List[A]
      * ---------------------------------------------------
      * object List filter/filter_1/filter_2/filter_dismantling函数
      */


    List.filter(List(1, 2, 3, 4, 5))(_ % 2 == 0)

    /** Exercise3.20
      * 写一个flatMap函数,它跟map函数有些像,除了传入的函数f返回的是列表而非单个结果.
      * 这个f所返回的列表会被塞到flatMap最终返回的列表.签名如下:
      * def flatMap[A,B](as:List[A])(f:A=>List[B]):List[B]
      * 例如:flatMap(List(1,2,3))(i=>List(i,i))结果是List(1,1,2,2,3,3)
      * --------------------------------------------------------------------
      * object List flatMap函数
      */

    List.flatMap(List(1, 2, 3))(i => List(i, i))

    /** Exercise3.21
      * 用flatMap实现filter
      * --------------------------------------------------------------------
      * object List filterViaFlatMap函数
      */

    /** Exercise3.22
      * 写一个函数,接收2歌列表,通过对相应元素的相加构造出一个新的列表.
      * 比如,List(1,2,3)和List(4,5,6)构造出List(5,7,9)
      * -------------------------------------------------------
      * object List addPairwise函数
      */

    /** Exercise3.23
      * 对刚才的函数泛化,不只针对整数或相加操作.
      * 将这个泛化函数命名为zipWith
      * -------------------------------------------------------
      * object List zipWith函数
      */

    /** Exercise3.24 <难>
      * 实现hasSubsequence方法,检测一个List子序列是否包含另一个List,
      * 比如,List(1,2,3,4)包含的子序列有List(1,2),List(2,3),List(4)等.
      * 或许找到一种简洁高效的纯函数式实现有些困难,没关系,先用最自然的想法实现,
      * 我们在第5章之后再回顾,到时候再改进这个实现.注意:任意两个值x和y在scala中
      * 可以是用表达式x==y来比较他们是否相等.
      * def hasSubsequence[A](sup:List[A],sub:List[A]):Boolean
      * -------------------------------------------------------
      * object List hasSubsequence函数
      */

    /** Exercise3.25
      * 写一个size函数,统计一颗树中的节点数(叶子节点和分支节点).
      * -------------------------------------------------------
      * object Tree 中的size函数
      */

    /** Exercise3.26
      * 写一个maximum函数,返回Tree[Int]中最大的元素(提示,在scala中使用
      * x.max(y)或x max y计算整数x和y的最大值).
      * -------------------------------------------------------
      * object Tree 中的maximum函数
      */

    /** Exercise3.27
      * 写一个depth函数,返回一棵树中从跟节点到任何叶子节点最大的路径长度.
      * -------------------------------------------------------
      * object Tree 中的depth函数
      */

    /** Exercise3.28
      * 写一个map函数,类似于List中的同名函数,接收一个函数,对树中每个元素进行修改.
      * -------------------------------------------------------
      * object Tree 中的map函数
      */
}

/** list 部分 */

sealed trait List[+A]

case object Nil extends List[Nothing]

case class Cons[+A](head: A, tail: List[A]) extends List[A]

object List {
    def sum(ints: List[Int]): Int = ints match {
        case Nil => 0
        case Cons(x, xs) => x + sum(xs)
    }

    def product(ds: List[Double]): Double = ds match {
        case Nil => 1.0
        case Cons(0.0, _) => 0.0
        case Cons(x, xs) => x * product(xs)
    }

    def apply[A](as: A*): List[A] =
        if (as.isEmpty) Nil
        else Cons(as.head, apply(as.tail: _*))


    def tail[A](list: List[A]): List[A] = {
        list match {
            case Nil => sys.error(s"tail of empty list")
            case Cons(_, t) => t
        }
    }

    def setHead[A](list: List[A], h: A): List[A] = {
        list match {
            case Nil => sys.error(s"setHead of empty list")
            case Cons(oh, t) => Cons(h, t)
        }
    }

    @tailrec
    def drop[A](l: List[A], n: Int): List[A] = {
        if (n <= 0) l
        else l match {
            case Nil => Nil
            case Cons(_, t) => drop(t, n - 1)
        }
    }

    def dropWhile[A](l: List[A], f: A => Boolean): List[A] = {
        l match {
            case Cons(h, t) if f(h) => dropWhile(t, f)
            case t => t
        }
    }

    def init[A](l: List[A]): List[A] = {
        l match {
            case Nil => sys.error(s"init of empty list")
            case Cons(_, Nil) => Nil
            case Cons(h, t) => Cons(h, init(t))
        }
    }

    def foldRight[A, B](as: List[A], z: B)(f: (A, B) => B): B = {
        as match {
            case Nil => z
            case Cons(x, xs) => f(x, foldRight(xs, z)(f))
        }
    }

    def sum2(ns: List[Int]) =
        foldRight(ns, 0)((x, y) => x + y)

    def product2(ns: List[Double]) =
        foldRight(ns, 1.0)(_ * _)

    def length[A](as: List[A]): Int = foldRight(as, 0)((_, acc) => acc + 1)

    @tailrec
    def foldLeft[A, B](as: List[A], z: B)(f: (B, A) => B): B = {
        as match {
            case Nil => z
            case Cons(x, xs) => foldLeft(xs, f(z, x))(f)
        }
    }

    def sum3(l: List[Int]): Int = foldLeft(l, 0)(_ + _)

    def product3(l: List[Double]): Double = foldLeft(l, 1.0)(_ * _)

    def length2[A](list: List[A]): Int = foldLeft(list, 0)((acc, _) => acc + 1)

    def reverse[A](list: List[A]): List[A] = foldLeft(list, List[A]())((acc, h) => Cons(h, acc))

    def foldRightViaFoldLeft[A, B](list: List[A], z: B)(f: (A, B) => B): B = foldLeft(reverse(list), z)((b, a) => f(a, b))

    def foldRightViaFoldLeft_1[A, B](list: List[A], z: B)(f: (A, B) => B): B = {
        //TODO 这有点难理解
        val part1: (((B) => B, A) => (B) => B) => (B) => B = foldLeft(list, (b: B) => b) //1.首先将foldLif中的第二个参数替换成一个(b: B) => b的函数
        // 这样就产生了一个需要接受一个(B) => B和一个A两个类型的参数的函数返回一额B的结果
        val part2: (B) => B = part1((g, a) => b => g(f(a, b))) // 这里 g 是一个 (B)=>B的函数 由于f(a,b)本身产生一个B类型的参数因此该函数符合(((B) => B, A) => (B) => B)
        //产生了一个(B) => B的函数
        val part3: B = part2(z) // 最后传入一个B
        //TODO 这里虽然能看懂,但是并不能理解作者的思维模式
        //foldLeft(list, (b: B) => b)((g, a) => b => g(f(a, b)))(z)  简化写法
        part3
    }

    def foldLeftViaFoldRight[A, B](list: List[A], z: B)(f: (B, A) => B): B = {
        //1.将B 替换为 (B)=>B 接受一个B返回一个B的函数

        val def1: (B) => B = (u: B) => u // 这里为了清晰将这个(B)=>B的函数定义为def1

        //2.我们将(B)=>B (这里即def1)带入B 得到part1
        val part1: ((A, (B) => B) => (B) => B) => (B) => B = foldRight(list, def1) // 到了这li part1是一个((A, (B) => B) => (B) => B) => (B) => B这样事儿的函数

        val def2: ((A, (B) => B) => (B) => B) = (a: A, g: (B) => B) => b => g(f(b, a)) // 这里为了清晰将这个  ((A, (B) => B) => (B) => B)的函数定义为def2
        //3.接下来我们将A,B函数带入
        val part2: (B) => B = part1(def2)
        //4.最后将z带入
        val part3 = part2(z)
        // 简化写法foldRight(l, (b:B) => b)((a,g) => b => g(f(b,a)))(z)
        part3
    }


    def append[A](a1: List[A], a2: List[A]): List[A] =
        a1 match {
            case Nil => a2
            case Cons(h, t) => Cons(h, append(t, a2))
        }

    def appendViaFoldRight[A](l: List[A], r: List[A]): List[A] = foldRight(l, r)(Cons(_, _))

    def appendViaFoldLeft[A](l: List[A], r: List[A]): List[A] = {
        val ml = l
        val mz = r
        val mF = (h: A, t: List[A]) => Cons(h, t)
        foldLeft(ml, (b: List[A]) => b)((g, a) => b => g(mF(a, b)))(mz)
    }

    def concat[A](l: List[List[A]]): List[A] = foldRight(l, Nil: List[A])(append)

    def add1(l: List[Int]): List[Int] = foldRight(l, Nil: List[Int])((h, t) => Cons(h + 1, t))

    def doubleToString(l: List[Double]): List[String] = foldRight(l, Nil: List[String])((h, t) => Cons(h.toString, t))

    def map[A, B](as: List[A])(f: A => B): List[B] = foldRight(as, Nil: List[B])((h, t) => Cons(f(h), t))

    def map_1[A, B](as: List[A])(f: A => B): List[B] = foldRightViaFoldLeft(as, Nil: List[B])((h, t) => Cons(f(h), t))

    def map_me[A, B](as: List[A])(f: A => B): List[B] = {
        val l = as
        val z = Nil: List[B]
        val tempF = (h: A, t: List[B]) => Cons(f(h), t)
        foldLeft(l, (b: List[B]) => b)((g, a) => b => g(tempF(a, b)))(z)
    }

    def map_2[A, B](as: List[A])(f: A => B): List[B] = {
        //这个感觉好弱智
        val buf = new scala.collection.mutable.ListBuffer[B]
        def go(l: List[A]): Unit = l match {
            case Nil => ()
            case Cons(h, t) => buf += f(h); go(t)
        }
        go(as)
        List(buf.toList: _*)
    }

    def filter[A](as: List[A])(f: A => Boolean): List[A] = foldRight(as, Nil: List[A])((h, t) => if (f(h)) Cons(h, t) else t)

    def filter_1[A](as: List[A])(f: A => Boolean): List[A] = foldRightViaFoldLeft_1(as, Nil: List[A])((h, t) => if (f(h)) Cons(h, t) else t)

    def filter_dismantling[A](as: List[A])(f: A => Boolean): List[A] = foldLeft(as, (b: List[A]) => b)((g, a) => b => g(if (f(a)) Cons(a, b) else b))(Nil: List[A])

    def filter_2[A](as: List[A])(f: A => Boolean): List[A] = {
        val buf = new scala.collection.mutable.ListBuffer[A]
        def go(l: List[A]): Unit = l match {
            case Nil => ()
            case Cons(h, t) => if (f(h)) buf += h; go(t)
        }
        go(as)
        List(buf.toList: _*)
    }

    def flatMap[A, B](as: List[A])(f: A => List[B]): List[B] = concat(map(as)(f))

    def filterViaFlatMap[A](as: List[A])(f: A => Boolean): List[A] = flatMap(as)(h => if (f(h)) List(h) else Nil)

    def addPairwise(a: List[Int], b: List[Int]): List[Int] = (a, b) match {
        case (Nil, _) => Nil
        case (_, Nil) => Nil
        case (Cons(h1, t1), Cons(h2, t2)) => Cons(h1 + h2, addPairwise(t1, t2))
    }

    def zipWith[A, B, C](a: List[A], b: List[B])(f: (A, B) => C): List[C] = (a, b) match {
        case (Nil, _) => Nil
        case (_, Nil) => Nil
        case (Cons(h1, t1), Cons(h2, t2)) => Cons(f(h1, h2), zipWith(t1, t2)(f))
    }

    @tailrec
    def startsWith[A](l: List[A], prefix: List[A]): Boolean = (l, prefix) match {
        case (_, Nil) => true
        case (Cons(h1, t1), Cons(h2, t2)) if h1 == h2 => startsWith(t1, t2)
        case _ => false
    }

    @tailrec
    def hasSubsequence[A](sup: List[A], sub: List[A]): Boolean = sup match {
        case Nil => sub == Nil
        case _ if startsWith(sup, sub) => true
        case Cons(h, t) => hasSubsequence(t, sub)
    }
}

/** tree 部分 */

sealed trait Tree[+A]

case class Leaf[A](value: A) extends Tree[A]

case class Branch[A](left: Tree[A], right: Tree[A]) extends Tree[A]

object Tree {

    def size[A](tree: Tree[A]): Int = tree match {
        case Leaf(n) => 1
        case Branch(l, r) => 1 + size(l) + size(r)
    }

    def maximum(tree: Tree[Int]): Int = tree match {
        case Leaf(n) => n
        case Branch(l, r) => maximum(l) max maximum(r)
    }

    def depth[A](tree: Tree[A]): Int = tree match {
        case Leaf(_) => 0
        case Branch(l, r) => 1 + (depth(l) max depth(r))
    }


    def map[A, B](tree: Tree[A])(f: A => B): Tree[B] = tree match {
        case Leaf(a) => Leaf(f(a))
        case Branch(l, r) => Branch(map(l)(f), map(l)(f))
    }

}