//! Free variable capture analysis.
//!
//! This module analyzes Python AST elements (like inner functions or lambdas) to detect
//! references to free variables defined in outer scopes (closures).

use rustpython_ast as ast;
use rustpython_ast::Visitor;
use std::collections::HashSet;

/// AST visitor to detect variable captures (non-local free variables).
///
/// Traverses expressions and statements to identify variable reads
/// that are not locally defined or passed as arguments.
pub struct CaptureVisitor {
    /// Collected names of captured variables.
    pub captures: HashSet<String>,
    /// Tracked variables that have been defined or bound within the current local scope.
    pub defined: HashSet<String>,
}

impl CaptureVisitor {
    /// Creates a new `CaptureVisitor` initialized with function parameters as defined names.
    pub fn new(params: Vec<String>) -> Self {
        let mut defined = HashSet::new();
        for p in params {
            defined.insert(p);
        }
        Self {
            captures: HashSet::new(),
            defined,
        }
    }
}


impl Visitor for CaptureVisitor {
    fn visit_expr(&mut self, expr: ast::Expr) {
        match &expr {
            ast::Expr::Name(n) => {
                if n.ctx.is_load() && !self.defined.contains(n.id.as_str()) {
                    self.captures.insert(n.id.to_string());
                } else if n.ctx.is_store() {
                    self.defined.insert(n.id.to_string());
                }
            }
            ast::Expr::Lambda(l) => {
                // Nested lambda:
                // 1. Collect all names used in this lambda
                let mut inner_visitor = CaptureVisitor::new(
                    l.args.args.iter().map(|a| a.def.arg.to_string()).collect(),
                );
                inner_visitor.visit_expr(*l.body.clone());

                // 2. Any names used in the inner lambda but NOT defined there
                // are potential captures for OUR lambda (if we don't define them either)
                for cap in inner_visitor.captures {
                    if !self.defined.contains(&cap) {
                        self.captures.insert(cap);
                    }
                }
                return;
            }
            _ => {}
        }
        self.generic_visit_expr(expr);
    }

    fn visit_stmt(&mut self, stmt: ast::Stmt) {
        match &stmt {
            ast::Stmt::Assign(a) => {
                for target in &a.targets {
                    self.visit_expr(target.clone());
                }
                self.visit_expr(*a.value.clone());
                return;
            }
            ast::Stmt::FunctionDef(f) => {
                // Nested function:
                // 1. Define the function name in current scope
                self.defined.insert(f.name.to_string());

                // 2. Analyze inner scope
                let inner_params: Vec<String> = f.args.args.iter().map(|a| a.def.arg.to_string()).collect();
                let mut inner_visitor = CaptureVisitor::new(inner_params);
                for s in &f.body {
                    inner_visitor.visit_stmt(s.clone());
                }

                // 3. Captures from inner scope that are NOT defined in inner scope
                // become potential captures for OUR scope.
                for cap in inner_visitor.captures {
                    if !self.defined.contains(&cap) {
                        self.captures.insert(cap);
                    }
                }
                return;
            }
            _ => {}
        }
        self.generic_visit_stmt(stmt);
    }
}
