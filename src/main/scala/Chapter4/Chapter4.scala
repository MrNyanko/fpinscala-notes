package Chapter4

/**
  * Created by Nyankosensei on 16/5/9.
  */
object Chapter4 extends App {

    /** Exercise4.1
      * 对Option实现之前的所有函数,在实现每一个函数时试着考虑它有什么意义,
      * 在什么场景下使用.接下来会探索何时使用这些函数.这里对解决这个联系
      * 给出一些提示:
      * •使用模式匹配也可以,虽然可以不借助模式匹配实现这些方法,初map和getOrElse.
      * •对map和flatMap,类型签名应该足以确定它的实现.
      * •getOrElse返回结果是Option类型,如果Option为None返回给定的默认值,否者结果被分装在Some对象里
      * •如果定义了的话,orElse返回第一个Option;否则返回第二个Option.
      * -----------------------------------------------------------------------------------
      * trait Option 中的map/getOrElse/flatMap/flatMap_1/orElse/orElse_1/filter/filter_1函数
      */

    /** Exercise4.2
      * 根据flatMap实现一个variance(方差)函数.如果一个序列的平均值是m,
      * variance是对序列中的每一个元素x进行math.pow(x-m,2).
      * 方差是各个数据与平均数之差的平方的平均数(初中定义)
      * def variance(xs: Seq[Double]): Option[Double]
      * ---------------------------------------------------------
      * object Option 中的variance函数
      */

    /** Exercise4.3
      * 写一个泛型函数map2,使用一个二元函数来组合两个Option值.如果两个Option
      * 都为None,也返回None.签名如下:
      * def map2[A, B, C](a: Option[A], b: Option[B])(f: (A, B) => C): Option[C]
      * ------------------------------------------------------------------------
      * object Option 中的map2函数
      */

    /** Exercise4.4
      * 写一个sequence函数,将一个Option列表结合为一个Option.这个结果
      * Option包含原Option列表中的所有元素(用Some封装的)值.如果原Option
      * 列表中出现一个None,函数结果也应该返回None;否则结果应该是所有(使用
      * Some包装的)元素值的列表.签名如下:
      * def sequence[A](a: List[Option[A]]): Option[List[A]]
      * ----------------------------------------------------------
      * object Option 中的sequence/sequence_1/sequence_me函数
      */

    /**
      * Exercise4.5
      * 实现一个函数,它直接使用map和sequence,但效率更好,只便利一次列表.实际上,
      * 按照traverse来实现sequence.
      * ---------------------------------------------------------------
      * object Option 中的sequenceViaTraverse函数
      */

    /** Exercise4.6
      * 实现Either版本的map,flatMap,orElse和map2函数
      * --------------------------------------------
      * trait Either 中的map/flatMap/orElse/map2函数
      */

    /** Exercise4.7
      * 对Either实现sequence和traverse,如果遇到错误返回第一个错误。
      * --------------------------------------------------------------
      * object Either 中的traverse/traverse_1/traverse_me/sequence函数
      */
}


/** None部分 */

sealed trait Option[+A] {
    def map[B](f: A => B): Option[B] = this match {
        case None => None
        case Some(a) => Some(f(a))
    }

    def getOrElse[B >: A](default: => B): B = this match {
        case None => default
        case Some(a) => a
    }

    def flatMap[B](f: A => Option[B]): Option[B] = map(f) getOrElse None

    def flatMap_1[B](f: A => Option[B]): Option[B] = this match {
        case None => None
        case Some(a) => f(a)
    }

    def orElse[B >: A](ob: => Option[B]): Option[B] = this map (Some(_)) getOrElse ob

    def orElse_1[B >: A](ob: => Option[B]): Option[B] = this match {
        case None => None
        case Some(a) => this
    }

    def filter(f: A => Boolean): Option[A] = this match {
        case Some(a) if f(a) => this
        case _ => None
    }

    def filter_1(f: A => Boolean): Option[A] = flatMap(a => if (f(a)) Some(a) else None)
}

case class Some[+A](get: A) extends Option[A]

case object None extends Option[Nothing]

object Option {

    def failingFn(i: Int): Int = {
        // `val y: Int = ...` declares `y` as having type `Int`, and sets it equal to the right hand side of the `=`.
        val y: Int = throw new Exception("fail!")
        try {
            val x = 42 + 5
            x + y
        }
        // A `catch` block is just a pattern matching block like the ones we've seen. `case e: Exception` is a pattern
        // that matches any `Exception`, and it binds this value to the identifier `e`. The match returns the value 43.
        catch {
            case e: Exception => 43
        }
    }

    def failingFn2(i: Int): Int = {
        try {
            val x = 42 + 5
            // A thrown Exception can be given any type; here we're annotating it with the type `Int`
            x + ((throw new Exception("fail!")): Int)
        }
        catch {
            case e: Exception => 43
        }
    }

    def mean(xs: Seq[Double]): Option[Double] =
        if (xs.isEmpty) None
        else Some(xs.sum / xs.length)

    def variance(xs: Seq[Double]): Option[Double] = mean(xs) flatMap { m => mean(xs map { x => math.pow(x - m, 2) }) }


    def map2[A, B, C](a: Option[A], b: Option[B])(f: (A, B) => C): Option[C] = a flatMap (aa => b map (bb => f(aa, bb)))


    def sequence[A](a: List[Option[A]]): Option[List[A]] = a match {
        case Nil => Some(Nil)
        case head :: tail => head flatMap (hh => sequence(tail) map (hh :: _))
    }


    def sequence_1[A](a: List[Option[A]]): Option[List[A]] =
        a.foldRight[Option[List[A]]](Some(Nil))((x, y) => map2(x, y)(_ :: _))

    def sequence_me[A](a: List[Option[A]]): Option[List[A]] = (a :\ (Some(Nil): Option[List[A]])) { (last, n) =>
        last flatMap (l => n.map(nn => l :: nn))
    }

    def traverse[A, B](a: List[A])(f: A => Option[B]): Option[List[B]] =
        a match {
            case Nil => Some(Nil)
            case head :: tail => map2(f(head), traverse(tail)(f))(_ :: _)
        }

    def traverse_1[A, B](a: List[A])(f: A => Option[B]): Option[List[B]] =
        a.foldRight[Option[List[B]]](Some(Nil))((h, t) => map2(f(h), t)(_ :: _))

    def sequenceViaTraverse[A](a: List[Option[A]]): Option[List[A]] = traverse(a)(x => x)

}

/** Either */

sealed trait Either[+E, +A] {
    def map[B](f: A => B): Either[E, B] =
        this match {
            case Right(a) => Right(f(a))
            case Left(e) => Left(e)
        }

    def flatMap[EE >: E, B](f: A => Either[EE, B]): Either[EE, B] =
        this match {
            case Left(e) => Left(e)
            case Right(a) => f(a)
        }


    def orElse[EE >: E, AA >: A](b: => Either[EE, AA]): Either[EE, AA] =
        this match {
            case Left(_) => b
            case Right(a) => Right(a)
        }

    def map2[EE >: E, B, C](b: Either[EE, B])(f: (A, B) => C): Either[EE, C] =
        for {a <- this; b1 <- b} yield f(a, b1)
}

case class Left[+E](value: E) extends Either[E, Nothing]

case class Right[+A](value: A) extends Either[Nothing, A]

object Either {
    def mean(xs: IndexedSeq[Double]): Either[String, Double] =
        if (xs.isEmpty)
            Left("mean of empty list")
        else
            Right(xs.sum / xs.length)

    def safeDiv(x: Int, y: Int): Either[Exception, Int] =
        try Right(x / y)
        catch {
            case e: Exception => Left(e)
        }

    def Try[A](a: => A): Either[Exception, A] =
        try Right(a)
        catch {
            case e: Exception => Left(e)
        }

    def traverse[E, A, B](es: List[A])(f: A => Either[E, B]): Either[E, List[B]] =
        es match {
            case Nil => Right(Nil)
            case h :: t => (f(h) map2 traverse(t)(f)) (_ :: _)
        }

    def traverse_1[E, A, B](es: List[A])(f: A => Either[E, B]): Either[E, List[B]] =
        es.foldRight[Either[E, List[B]]](Right(Nil))((a, b) => f(a).map2(b)(_ :: _))

    def traverse_me[E, A, B](es: List[A])(f: A => Either[E, B]): Either[E, List[B]] =
        (es :\ (Right(Nil): Either[E, List[B]])) { (a, b) => f(a).map2(b)(_ :: _) }

    def sequence[E, A](es: List[Either[E, A]]): Either[E, List[A]] =
        traverse(es)(x => x)
}