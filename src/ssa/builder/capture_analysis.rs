use rustpython_ast as ast;
use rustpython_ast::Visitor;
use std::collections::HashSet;

pub struct CaptureVisitor {
    pub captures: HashSet<String>,
    pub defined: HashSet<String>,
}

impl CaptureVisitor {
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
            _ => {}
        }
        self.generic_visit_stmt(stmt);
    }
}
