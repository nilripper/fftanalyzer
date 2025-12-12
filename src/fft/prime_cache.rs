use lazy_static::lazy_static;
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static! {
    //
    // Global PrimeLore instance protected by a mutex.
    //
    static ref PRIME_LORE: Mutex<PrimeLore> = Mutex::new(PrimeLore::new());
}

struct PrimeLore {
    smallest_factors: HashMap<usize, usize>,
    primes: Vec<usize>, // Ordered list of known primes.
    last_prime: usize,
}

impl PrimeLore {
    fn new() -> Self {
        let mut s = Self {
            smallest_factors: HashMap::from([(0, 0), (1, 1), (2, 2)]),
            primes: vec![2],
            last_prime: 2,
        };

        //
        // Warm the cache with initial factorization results.
        //
        for i in 0..1024 {
            s.find(i);
        }
        s
    }

    fn find(&mut self, n: usize) -> usize {
        //
        // Return cached result if available.
        //
        if let Some(&f) = self.smallest_factors.get(&n) {
            return f;
        }

        let mut solution = n;

        //
        // Check divisibility using already known primes (ordered).
        //
        for &p in &self.primes {
            if p * p > n {
                break;
            }
            if n % p == 0 {
                solution = p;
                break;
            }
        }

        //
        // Continue searching for a divisor using odd candidates.
        //
        if solution == n {
            let mut p = self.last_prime | 1;
            while p * p <= n {
                if n % p == 0 {
                    solution = p;
                    break;
                }
                p += 2;
            }
        }

        //
        // If no divisor was found, n is prime; update prime list.
        //
        if solution == n {
            if n > self.last_prime {
                self.primes.push(n);
                self.last_prime = n;
            } else {
                if let Err(pos) = self.primes.binary_search(&n) {
                    self.primes.insert(pos, n);
                }
            }
        }

        //
        // Cache the smallest factor for n.
        //
        self.smallest_factors.insert(n, solution);
        solution
    }
}

pub fn get_factors_all(mut n: usize) -> (Vec<usize>, usize) {
    let mut factors = Vec::with_capacity(16);
    let mut count = 0;

    let mut lore = PRIME_LORE.lock().unwrap();

    //
    // Return empty result for 0 or 1.
    //
    if n <= 1 {
        return (factors, 0);
    }

    loop {
        let f = lore.find(n);
        factors.push(f);
        count += 1;

        //
        // If n equals its smallest factor, factorization is complete.
        //
        if f == n {
            break;
        }
        n /= f;
    }
    (factors, count)
}
