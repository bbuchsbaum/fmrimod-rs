# Python to Rust Library Conversion - Claude Code Agent Guide

## Core Objective
Convert Python packages to Rust with maximum performance, safety, and idiomatic Rust design. Produce Rust libraries that maintain API compatibility where sensible while leveraging Rust's ownership system, type safety, and zero-cost abstractions.

## Phase 0: Discovery & Assessment

### Initial Codebase Analysis
1. **Repository Scan**
   - Run `pip install -e .` and `python -m pytest` to identify existing issues
   - Use `ast` module to parse and analyze Python code structure
   - Create dependency tree with `pipdeptree` or analyze `requirements.txt`/`pyproject.toml`
   - Tag dependencies as stdlib vs PyPI vs local

2. **Complexity Assessment**
   - For each module/class/function, assign translation risk:
     - **High**: Dynamic typing abuse, metaclasses, monkey patching, heavy reflection
     - **Medium**: Complex inheritance, operator overloading, context managers
     - **Low**: Pure functions, simple classes, basic I/O

3. **Dynamic Features Inventory**
   ```bash
   # Find dynamic attribute access
   rg -n "(getattr|setattr|delattr|__getattribute__|__dict__|exec|eval)" --type py | tee dynamic_features.txt
   
   # Find metaclasses and decorators
   rg -n "(type\(|metaclass=|@\w+)" --type py | tee metaclass_decorator_usage.txt
   ```

4. **Baseline Testing**
   - Extract existing pytest/unittest tests
   - Document test coverage with `pytest --cov`
   - Record performance baselines with `timeit` or `pytest-benchmark`
   - Note any randomness/nondeterminism

## Phase 1: Architecture Mapping

### Python to Rust Concept Translation

| Python Concept | Rust Equivalent | Implementation Notes |
|----------------|-----------------|---------------------|
| Class with `__init__` | `struct` with `impl` and `new()` | Constructor pattern |
| `@dataclass` | `#[derive(Debug, Clone)]` struct | Add serde for serialization |
| ABC/Protocol | Trait | Define behavior contracts |
| Instance methods | `impl` block methods | `&self`, `&mut self`, or `self` |
| Class methods | Associated functions | No `self` parameter |
| Properties | Getter/setter methods | Or public fields if simple |
| `__str__`/`__repr__` | `Display`/`Debug` traits | Implement formatting |
| Context managers | RAII or `Drop` trait | Automatic cleanup |
| Generators | Iterator trait | Lazy evaluation |
| Async/await | `async`/`await` with tokio | Match runtime choice |
| Exception handling | `Result<T, E>` | Explicit error handling |
| `None` | `Option<T>` | Null safety |
| Dynamic typing | Generics or enums | Type safety first |
| Global state | `lazy_static!` or `once_cell` | Minimize usage |
| Metaclasses | Build-time code generation | Macros if necessary |

### Memory Model Mapping
| Python Pattern | Rust Approach | Key Consideration |
|----------------|---------------|-------------------|
| Reference counting | Ownership | Single owner by default |
| Shared references | `Rc<T>` or `Arc<T>` | Choose based on threading |
| Mutable aliasing | `RefCell<T>` or `Mutex<T>` | Interior mutability |
| Weak references | `Weak<T>` | Prevent cycles |
| Object pools | Arena allocators | Consider `typed-arena` |

### Error Handling Strategy
```rust
// Define custom error types
#[derive(Debug, thiserror::Error)]
pub enum MyError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// Python: raise ValueError("message")
// Rust: return Err(MyError::InvalidInput("message".into()));
```

## Phase 2: Implementation Strategy

### Project Structure
```
crate_name/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs          # Public API
│   ├── error.rs        # Error types
│   ├── types.rs        # Core types
│   ├── core/
│   │   └── mod.rs
│   └── utils/
│       └── mod.rs
├── tests/
│   ├── integration/
│   └── fixtures/
├── benches/
│   └── benchmarks.rs
├── examples/
└── python/             # Python bindings (if needed)
    └── src/
```

### Cargo.toml Template
```toml
[package]
name = "crate_name"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <email@example.com>"]
description = "Rust implementation of Python's X library"
license = "MIT OR Apache-2.0"
repository = "https://github.com/user/repo"

[dependencies]
thiserror = "1.0"
serde = { version = "1.0", features = ["derive"], optional = true }
rayon = { version = "1.7", optional = true }

[dev-dependencies]
criterion = "0.5"
proptest = "1.2"
approx = "0.5"

[features]
default = []
serde = ["dep:serde"]
parallel = ["dep:rayon"]

[[bench]]
name = "benchmarks"
harness = false
```

### Migration Loop Pattern
1. **Start with core types** (data structures)
2. **Implement pure functions** (no side effects)
3. **Add methods and traits** (behavior)
4. **Handle I/O and errors** (Result types)
5. **Optimize hot paths** (benchmarks first)
6. **Add Python bindings** (PyO3 if needed)

### Implementation Checklist
- [ ] Read Python source and tests thoroughly
- [ ] Design Rust API (may differ from Python)
- [ ] Implement with strong typing
- [ ] Use `clippy` and `rustfmt`
- [ ] Document with examples
- [ ] Benchmark against Python
- [ ] Fuzz test with `cargo-fuzz`

## Phase 3: Type System Translation

### Common Type Mappings

| Python Type | Rust Type | Notes |
|-------------|-----------|-------|
| `int` | `i64` or `i32` | Check range requirements |
| `float` | `f64` | Consider `f32` for performance |
| `str` | `String` or `&str` | Ownership matters |
| `bytes` | `Vec<u8>` or `&[u8]` | Mutable vs immutable |
| `list` | `Vec<T>` | Homogeneous types |
| `tuple` | `(T1, T2, ...)` | Fixed size |
| `dict` | `HashMap<K, V>` | Or `BTreeMap` if ordered |
| `set` | `HashSet<T>` | Or `BTreeSet` if ordered |
| `typing.Optional[T]` | `Option<T>` | Explicit null handling |
| `typing.Union[T1, T2]` | `enum` | Algebraic data types |
| `typing.List[T]` | `Vec<T>` | Type parameter |
| `typing.Dict[K, V]` | `HashMap<K, V>` | Generic parameters |
| `typing.Callable` | `Fn` traits | Closure traits |
| NumPy array | `ndarray::Array` | Or custom types |
| Pandas DataFrame | Custom struct | With `polars` or similar |

### Pattern Translation

#### List Comprehensions
```python
# Python
squares = [x**2 for x in range(10) if x % 2 == 0]
```
```rust
// Rust
let squares: Vec<i32> = (0..10)
    .filter(|x| x % 2 == 0)
    .map(|x| x * x)
    .collect();
```

#### Dictionary Comprehensions
```python
# Python
word_lengths = {word: len(word) for word in words}
```
```rust
// Rust
let word_lengths: HashMap<String, usize> = words
    .into_iter()
    .map(|word| (word.clone(), word.len()))
    .collect();
```

#### Context Managers
```python
# Python
with open("file.txt") as f:
    content = f.read()
```
```rust
// Rust (automatic with RAII)
let content = std::fs::read_to_string("file.txt")?;

// Or for custom resources:
{
    let file = File::open("file.txt")?;
    // file automatically closed when out of scope
}
```

## Phase 4: Testing Strategy

### Testing Layers

| Layer | Rust Approach | Success Criterion |
|-------|---------------|-------------------|
| Unit tests | `#[cfg(test)] mod tests` | All pass with `cargo test` |
| Property tests | `proptest` | Invariants hold |
| Benchmarks | `criterion` | Meet performance goals |
| Integration tests | `tests/` directory | API works as expected |
| Fuzzing | `cargo-fuzz` | No panics or UB |
| Miri | `cargo +nightly miri test` | No undefined behavior |

### Cross-Testing Infrastructure
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  test:
    strategy:
      matrix:
        rust: [stable, beta, nightly]
        os: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v3
      - uses: dtolnay/rust-toolchain@stable
        with:
          toolchain: ${{ matrix.rust }}
          components: rustfmt, clippy
      - run: cargo fmt --check
      - run: cargo clippy -- -D warnings
      - run: cargo test
      - run: cargo bench --no-run
```

### Python Compatibility Testing
```rust
// tests/python_compat.rs
#[test]
fn test_compatibility_with_python() {
    // Load Python test fixtures
    let input: serde_json::Value = 
        serde_json::from_str(include_str!("fixtures/input.json")).unwrap();
    
    // Run Rust implementation
    let rust_output = my_function(input);
    
    // Load Python output
    let python_output: serde_json::Value = 
        serde_json::from_str(include_str!("fixtures/python_output.json")).unwrap();
    
    // Compare
    assert_eq!(rust_output, python_output);
}
```

## Phase 5: Performance Optimization

### Optimization Priority
1. **Profile first** with `cargo flamegraph` or `perf`
2. **Use iterators** instead of collecting intermediates
3. **Leverage `rayon`** for easy parallelization
4. **Consider `unsafe`** only for proven bottlenecks
5. **Use const generics** for compile-time optimization
6. **SIMD** via `std::simd` or `packed_simd`

### Memory Optimization
```rust
// Use Cow for potentially borrowed data
use std::borrow::Cow;

fn process_string(s: &str) -> Cow<str> {
    if needs_processing(s) {
        Cow::Owned(s.to_uppercase())
    } else {
        Cow::Borrowed(s)
    }
}

// Use SmallVec for small collections
use smallvec::SmallVec;
type SmallString = SmallVec<[char; 16]>;
```

### Zero-Copy Patterns
```rust
// Python: data processing with copies
// Rust: borrowing and slicing
fn process_data(data: &[u8]) -> Result<ProcessedData, Error> {
    // Work with borrowed slices
    let header = &data[..HEADER_SIZE];
    let body = &data[HEADER_SIZE..];
    
    // Zero-copy parsing
    Ok(ProcessedData {
        header: Header::parse(header)?,
        body,  // Just store the slice
    })
}
```

## Phase 6: Documentation & Publishing

### Documentation Requirements
1. **Doc comments**: With examples for all public items
2. **README.md**: Quick start and feature overview
3. **MIGRATION.md**: Python to Rust migration guide
4. **API docs**: `cargo doc --open`

### Documentation Template
```rust
/// Computes the factorial of a number.
///
/// # Arguments
///
/// * `n` - A non-negative integer
///
/// # Returns
///
/// Returns `Some(factorial)` if computation succeeds, `None` on overflow.
///
/// # Examples
///
/// ```
/// use my_crate::factorial;
///
/// assert_eq!(factorial(5), Some(120));
/// assert_eq!(factorial(0), Some(1));
/// ```
///
/// # Python Equivalent
///
/// ```python
/// def factorial(n):
///     if n < 0:
///         raise ValueError("n must be non-negative")
///     result = 1
///     for i in range(2, n + 1):
///         result *= i
///     return result
/// ```
pub fn factorial(n: u32) -> Option<u128> {
    (1..=n as u128).try_fold(1u128, |acc, i| acc.checked_mul(i))
}
```

### Publishing Checklist
- [ ] Run `cargo fmt`
- [ ] Run `cargo clippy -- -D warnings`
- [ ] Run `cargo test`
- [ ] Run `cargo doc --no-deps`
- [ ] Update version in `Cargo.toml`
- [ ] Update `CHANGELOG.md`
- [ ] Tag release: `git tag -a v0.1.0 -m "Release v0.1.0"`
- [ ] Publish: `cargo publish --dry-run` then `cargo publish`

## Phase 7: Python Bindings (Optional)

### PyO3 Setup
```toml
# Cargo.toml
[dependencies]
pyo3 = { version = "0.19", features = ["extension-module"] }

[lib]
name = "my_crate"
crate-type = ["cdylib", "rlib"]
```

### Binding Implementation
```rust
use pyo3::prelude::*;

#[pyclass]
struct MyClass {
    #[pyo3(get, set)]
    value: i32,
}

#[pymethods]
impl MyClass {
    #[new]
    fn new(value: i32) -> Self {
        MyClass { value }
    }
    
    fn process(&self) -> PyResult<String> {
        Ok(format!("Value: {}", self.value))
    }
}

#[pymodule]
fn my_crate(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<MyClass>()?;
    Ok(())
}
```

## Critical Implementation Rules

### Always
- Think in terms of ownership and borrowing
- Handle all error cases explicitly
- Use strong typing - avoid `Box<dyn Any>`
- Run `clippy` on every change
- Benchmark before optimizing
- Document public APIs thoroughly
- Consider thread safety from the start

### Never
- Use `unwrap()` in library code (except tests)
- Ignore `clippy` warnings without good reason
- Use `unsafe` without proving safety
- Recreate Python's dynamic behavior exactly
- Add dependencies without evaluating alternatives
- Assume Python's GIL-based thread safety

## Common Patterns Reference

### Builder Pattern
```rust
// Python: Class with many optional parameters
// Rust: Builder pattern
#[derive(Default)]
pub struct ConfigBuilder {
    host: Option<String>,
    port: Option<u16>,
    timeout: Option<Duration>,
}

impl ConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = Some(host.into());
        self
    }
    
    pub fn build(self) -> Result<Config, Error> {
        Ok(Config {
            host: self.host.ok_or(Error::MissingHost)?,
            port: self.port.unwrap_or(8080),
            timeout: self.timeout.unwrap_or(Duration::from_secs(30)),
        })
    }
}
```

### Type State Pattern
```rust
// Encode state in the type system
pub struct Connection<State> {
    inner: TcpStream,
    _state: PhantomData<State>,
}

pub struct Disconnected;
pub struct Connected;

impl Connection<Disconnected> {
    pub fn connect(addr: &str) -> Result<Connection<Connected>, Error> {
        let inner = TcpStream::connect(addr)?;
        Ok(Connection {
            inner,
            _state: PhantomData,
        })
    }
}

impl Connection<Connected> {
    pub fn send(&mut self, data: &[u8]) -> Result<(), Error> {
        self.inner.write_all(data)?;
        Ok(())
    }
}
```

### Interior Mutability
```rust
// Python: Mutable class attributes
// Rust: Interior mutability when needed
use std::cell::RefCell;
use std::rc::Rc;

struct SharedState {
    data: Rc<RefCell<HashMap<String, String>>>,
}

impl SharedState {
    fn update(&self, key: String, value: String) {
        self.data.borrow_mut().insert(key, value);
    }
}
```

## Workflow Commands

### Initial Analysis
```bash
# Analyze Python structure
find . -name "*.py" -type f | head -20
grep -r "class\|def" --include="*.py" | head -20

# Check for complex Python features
rg "metaclass|__new__|__slots__|@property" --type py

# Identify test structure
find . -name "test_*.py" -o -name "*_test.py" | head -10
```

### Rust Project Setup
```bash
# Create new crate
cargo new --lib my_crate
cd my_crate

# Add common dependencies
cargo add thiserror
cargo add --dev criterion proptest

# Set up pre-commit hooks
echo 'cargo fmt --check && cargo clippy -- -D warnings && cargo test' > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### Development Workflow
```bash
# Watch for changes and test
cargo watch -x test

# Run benchmarks
cargo bench

# Generate and view docs
cargo doc --open

# Check for security issues
cargo audit

# Update dependencies
cargo update
```

## Success Metrics
- ✅ Core functionality ported
- ✅ Performance improvement demonstrated
- ✅ Memory safety guaranteed
- ✅ All tests passing
- ✅ Zero `unsafe` or justified with safety comments
- ✅ `clippy` warnings addressed
- ✅ Documentation complete
- ✅ Published to crates.io

## Agent-Specific Instructions

1. **Start with types**: Define core structs/enums first
2. **Think ownership**: Who owns data? Who borrows?
3. **Error early**: Define error types upfront
4. **Test continuously**: Write tests alongside implementation
5. **Benchmark often**: Ensure performance goals are met
6. **Document as you go**: Examples in doc comments
7. **Use Rust idioms**: Don't transliterate Python

## Common Pitfalls and Solutions

### String Handling
```rust
// Python: Strings are immutable but freely copied
// Rust: Be explicit about ownership

// Accepting strings
fn process(s: &str) { }  // Borrowed
fn store(s: String) { }   // Owned
fn flexible(s: impl Into<String>) { }  // Either

// Returning strings
fn get_borrowed() -> &'static str { "hello" }
fn get_owned() -> String { "hello".to_string() }
```

### Collection Iteration
```rust
// Python: for item in list:
// Rust: Multiple options

// Consuming iterator (takes ownership)
for item in vec {
    // vec cannot be used after this
}

// Borrowing iterator
for item in &vec {
    // item is &T
}

// Mutable borrowing iterator
for item in &mut vec {
    // item is &mut T
}
```

### Async Patterns
```rust
// Python: async def / await
// Rust: Choose runtime (tokio, async-std)

use tokio;

#[tokio::main]
async fn main() {
    let result = fetch_data().await;
}

async fn fetch_data() -> Result<Data, Error> {
    // Async operations
}
```

This prompt provides a complete guide for Python→Rust library conversion optimized for autonomous agent execution, maintaining the same structured approach as the R→Python guide while addressing Rust-specific concerns like ownership, type safety, and performance.