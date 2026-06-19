use super::types::Bound;

pub fn parse_refinement_bounds(ref_str: &str) -> (Bound, Bound) {
    if let Ok((l, h)) = eval_refinement_sexpr(ref_str.trim()) {
        (l, h)
    } else {
        (Bound::NegInf, Bound::PosInf)
    }
}

fn eval_refinement_sexpr(sexpr: &str) -> Result<(Bound, Bound), String> {
    if !sexpr.starts_with('(') {
        return Err("Not an S-expression".to_string());
    }

    let inner = &sexpr[1..sexpr.len() - 1];
    let parts = split_sexpr_parts(inner);
    if parts.is_empty() {
        return Err("Empty S-expression".to_string());
    }

    match parts[0] {
        "and" | "&" => {
            let mut low = Bound::NegInf;
            let mut high = Bound::PosInf;
            for part in &parts[1..] {
                if let Ok((l, h)) = eval_refinement_sexpr(part) {
                    low = low.max(l);
                    high = high.min(h);
                }
            }
            Ok((low, high))
        }
        "or" | "|" => {
            let mut low = Bound::PosInf;
            let mut high = Bound::NegInf;
            for part in &parts[1..] {
                if let Ok((l, h)) = eval_refinement_sexpr(part) {
                    low = low.min(l);
                    high = high.max(h);
                }
            }
            if low == Bound::PosInf || high == Bound::NegInf {
                Ok((Bound::NegInf, Bound::PosInf))
            } else {
                Ok((low, high))
            }
        }
        "=" | "==" => {
            if parts.len() != 3 {
                return Err("= expects 2 args".to_string());
            }
            let val = if parts[1] == "{v}" || parts[1] == "VALUE_PLACEHOLDER" {
                parts[2].parse::<f64>().ok()
            } else if parts[2] == "{v}" || parts[2] == "VALUE_PLACEHOLDER" {
                parts[1].parse::<f64>().ok()
            } else {
                None
            };
            if let Some(v) = val {
                Ok((Bound::Finite(v), Bound::Finite(v)))
            } else {
                Ok((Bound::NegInf, Bound::PosInf))
            }
        }
        "<" => {
            if parts.len() != 3 {
                return Err("< expects 2 args".to_string());
            }
            if parts[1] == "{v}" || parts[1] == "VALUE_PLACEHOLDER" {
                if let Ok(v) = parts[2].parse::<f64>() {
                    return Ok((Bound::NegInf, Bound::Finite(v - 1.0)));
                }
            } else if parts[2] == "{v}" || parts[2] == "VALUE_PLACEHOLDER" {
                if let Ok(v) = parts[1].parse::<f64>() {
                    return Ok((Bound::Finite(v + 1.0), Bound::PosInf));
                }
            }
            Ok((Bound::NegInf, Bound::PosInf))
        }
        "<=" => {
            if parts.len() != 3 {
                return Err("<= expects 2 args".to_string());
            }
            if parts[1] == "{v}" || parts[1] == "VALUE_PLACEHOLDER" {
                if let Ok(v) = parts[2].parse::<f64>() {
                    return Ok((Bound::NegInf, Bound::Finite(v)));
                }
            } else if parts[2] == "{v}" || parts[2] == "VALUE_PLACEHOLDER" {
                if let Ok(v) = parts[1].parse::<f64>() {
                    return Ok((Bound::Finite(v), Bound::PosInf));
                }
            }
            Ok((Bound::NegInf, Bound::PosInf))
        }
        ">" => {
            if parts.len() != 3 {
                return Err("> expects 2 args".to_string());
            }
            if parts[1] == "{v}" || parts[1] == "VALUE_PLACEHOLDER" {
                if let Ok(v) = parts[2].parse::<f64>() {
                    return Ok((Bound::Finite(v + 1.0), Bound::PosInf));
                }
            } else if parts[2] == "{v}" || parts[2] == "VALUE_PLACEHOLDER" {
                if let Ok(v) = parts[1].parse::<f64>() {
                    return Ok((Bound::NegInf, Bound::Finite(v - 1.0)));
                }
            }
            Ok((Bound::NegInf, Bound::PosInf))
        }
        ">=" => {
            if parts.len() != 3 {
                return Err(">= expects 2 args".to_string());
            }
            if parts[1] == "{v}" || parts[1] == "VALUE_PLACEHOLDER" {
                if let Ok(v) = parts[2].parse::<f64>() {
                    return Ok((Bound::Finite(v), Bound::PosInf));
                }
            } else if parts[2] == "{v}" || parts[2] == "VALUE_PLACEHOLDER" {
                if let Ok(v) = parts[1].parse::<f64>() {
                    return Ok((Bound::NegInf, Bound::Finite(v)));
                }
            }
            Ok((Bound::NegInf, Bound::PosInf))
        }
        _ => Ok((Bound::NegInf, Bound::PosInf)),
    }
}

fn split_sexpr_parts(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut current_start = 0;
    let mut depth = 0;
    let chars: Vec<char> = s.chars().collect();

    for i in 0..chars.len() {
        match chars[i] {
            '(' => depth += 1,
            ')' => depth -= 1,
            ' ' if depth == 0 => {
                let part = s[current_start..i].trim();
                if !part.is_empty() {
                    parts.push(part);
                }
                current_start = i + 1;
            }
            _ => {}
        }
    }
    let last_part = s[current_start..].trim();
    if !last_part.is_empty() {
        parts.push(last_part);
    }
    parts
}
