use mlab_rs::np;

fn main() {
    println!("--- NumPy Demo ---");

    // Python: a = np.array([1.0, 2.0, 3.0])
    let a = np::array(vec![1.0, 2.0, 3.0]);
    println!("a (1D Array):\n{:?}\n", a);

    // Python: b = np.zeros((2, 3))
    let b: np::Array2<f64> = np::zeros((2, 3));
    println!("b (2D Zeros Array):\n{:?}\n", b);

    // Python: c = np.arange(0, 10, 2)
    let c = np::arange(0.0, 10.0, 2.0);
    println!("c (Arange):\n{:?}\n", c);

    // Python: d = np.linspace(0, 1, 5)
    let d = np::linspace(0.0, 1.0, 5);
    println!("d (Linspace):\n{:?}\n", d);

    // Python: e = a + c[:3]
    // Slicing in Rust: use np::s! macro
    let e = &a + &c.slice(np::s![0..3]);
    println!("e (a + c[:3]):\n{:?}\n", e);

    // Python: print(np.mean(e))
    println!("mean(e): {}", np::mean(&e));
    println!("sum(e): {}", np::sum(&e));
}
