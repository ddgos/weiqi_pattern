use std::{num::ParseIntError, str::FromStr};

use vec2d::{Size, Vec2D};

pub enum Player {
    Black,
    White,
}

pub type Intersection = Option<Player>;

pub struct Edges {
    pub north: bool,
    pub east: bool,
    pub south: bool,
    pub west: bool,
}

pub struct Pattern {
    pub edges: Edges,
    pub pattern: Vec2D<Intersection>,
}

#[derive(Debug, PartialEq)]
pub enum PatternParseError {
    InvalidStructure,
    WidthParseError(ParseIntError),
    HeightParseError(ParseIntError),
    UnexpectedIntersection,
    BadSize,
}

impl FromStr for Pattern {
    type Err = PatternParseError;

    fn from_str(s: &str) -> Result<Self, PatternParseError> {
        let (edges_part, width_part, height_part, src_part) = {
            let mut parts = s.split(';');

            let edges_part = parts.next().ok_or(PatternParseError::InvalidStructure)?;
            let width_part = parts.next().ok_or(PatternParseError::InvalidStructure)?;
            let height_part = parts.next().ok_or(PatternParseError::InvalidStructure)?;
            let src_part = parts.next().ok_or(PatternParseError::InvalidStructure)?;
            (edges_part, width_part, height_part, src_part)
        };
        let edges = {
            let north = edges_part.contains('n');
            let east = edges_part.contains('e');
            let south = edges_part.contains('s');
            let west = edges_part.contains('w');
            Edges {
                north,
                east,
                south,
                west,
            }
        };
        let size = {
            let width = width_part
                .parse()
                .map_err(|e| PatternParseError::WidthParseError(e))?;
            let height = height_part
                .parse()
                .map_err(|e| PatternParseError::HeightParseError(e))?;
            Size { width, height }
        };
        let pattern = {
            let src = src_part
                .bytes()
                .map(|b| match b {
                    b'o' => Ok(Some(Player::White)),
                    b'x' => Ok(Some(Player::Black)),
                    b'.' => Ok(None),
                    _ => Err(PatternParseError::UnexpectedIntersection),
                })
                .collect::<Result<Vec<Intersection>, PatternParseError>>()?;
            Vec2D::from_vec(size, src).ok_or(PatternParseError::BadSize)?
        };

        Ok(Self { edges, pattern })
    }
}

impl Pattern {
    pub fn repr(&self) -> String {
        let mut s = String::with_capacity(self.size().area() + 11);
        if self.edges.north {
            s.push('n')
        };
        if self.edges.east {
            s.push('e')
        };
        if self.edges.south {
            s.push('s')
        };
        if self.edges.west {
            s.push('w')
        };
        s.push_str(&format!(";{};{};", self.width(), self.height()));
        self.pattern
            .iter()
            .map(|(_, intersection)| match intersection {
                Some(Player::Black) => 'x',
                Some(Player::White) => 'o',
                None => '.',
            })
            .for_each(|c| s.push(c));
        s
    }

    fn size(&self) -> Size {
        self.pattern.size()
    }

    fn width(&self) -> usize {
        self.size().width
    }

    fn height(&self) -> usize {
        self.size().height
    }
}

trait Rotate
where
    Self: Sized + Clone,
{
    fn rotate(&self, rotation: Rotation) -> Self {
        match rotation {
            Rotation::None => self.clone(),
            Rotation::Quarter => self.rotate_quarter(),
            Rotation::Half => self.rotate_half(),
            Rotation::ThreeQuarters => self.rotate_three_quarters(),
        }
    }

    fn rotate_quarter(&self) -> Self;

    fn rotate_half(&self) -> Self {
        self.rotate_quarter().rotate_quarter()
    }

    fn rotate_three_quarters(&self) -> Self {
        self.rotate_half().rotate_quarter()
    }
}

pub enum Rotation {
    None,
    Quarter,
    Half,
    ThreeQuarters,
}

pub struct PatternVariator {
    pub swap_colours: bool,
    pub reflect: bool,
    pub rotation: Rotation,
}

#[cfg(test)]
mod tests {
    use crate::{Pattern, PatternParseError};

    #[test]
    fn good_pattern_repr_works() {
        let pattern_str =
            concat!("ne;8;5;", "........", ".o......", "...oo...", "..xxx...", "........");
        let parsed_pattern: Pattern = pattern_str.parse().unwrap();
        let pattern_repr = parsed_pattern.repr();

        assert_eq!(pattern_str, pattern_repr);
    }

    #[test]
    fn bad_pattern_size_parse_fails() {
        let pattern_str = concat!(
            "nesw;7;4;",
            "........",
            ".o......",
            "...oo...",
            "..xxx...",
            "........"
        );
        let pattern_parse_result: Result<Pattern, _> = pattern_str.parse();

        assert!(pattern_parse_result.is_err_and(|e| e == PatternParseError::BadSize))
    }
}
