#[macro_export]
macro_rules! push_inst {
    ($builder:expr, $kind:expr) => {{
        let loc = $builder.current_location;
        let kind = $kind;
        let inst = $builder.add_instruction(kind);
        if let Some(l) = loc {
            inst.location = Some(l);
        }
        inst
    }};
    ($builder:expr, $kind:expr, $loc:expr) => {{
        let kind = $kind;
        let inst = $builder.add_instruction(kind);
        inst.location = Some($loc);
        inst
    }};
}

#[macro_export]
macro_rules! ensure_type {
    ($builder:expr, $val:expr, $expected:expr, $loc:expr) => {{
        let found = $builder.func.get_type($val);
        if found != $expected {
            return Err($crate::builder::error::BuilderError::TypeMismatch {
                expected: format!("{:?}", $expected),
                found: format!("{:?}", found),
                location: Some($loc),
            });
        }
    }};
}

#[macro_export]
macro_rules! builder_error {
    (TypeMismatch, $expected:expr, $found:expr) => {
        $crate::builder::error::BuilderError::TypeMismatch {
            expected: $expected,
            found: $found,
            location: None,
        }
    };
    (AttributeNotFound, $target:expr, $attr:expr) => {
        $crate::builder::error::BuilderError::AttributeNotFound {
            target: $target,
            attr: $attr,
            location: None,
        }
    };
    ($variant:ident, $loc:expr; $($arg:tt)*) => {
        $crate::builder::error::BuilderError::$variant(format!($($arg)*), Some($loc))
    };
    ($variant:ident, $($arg:tt)*) => {
        $crate::builder::error::BuilderError::$variant(format!($($arg)*), None)
    };
}
