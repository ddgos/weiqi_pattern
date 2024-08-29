use std::{num::ParseIntError, str::FromStr};

use vec2d::{Coord, Size, Vec2D};

#[derive(Debug, Clone, PartialEq)]
pub enum Player {
    Black,
    White,
}

pub type Intersection = Option<Player>;

#[derive(Clone, Debug, PartialEq, Copy)]
pub struct Edges {
    pub north: bool,
    pub east: bool,
    pub south: bool,
    pub west: bool,
}

impl Rotate for Edges {
    fn rotate_quarter(&self) -> Self {
        Self {
            north: self.west,
            east: self.north,
            south: self.east,
            west: self.south,
        }
    }
}

#[derive(PartialEq, Debug)]
pub struct Pattern {
    pub edges: Edges,
    pub pattern: Vec2D<Intersection>,
}

impl Clone for Pattern {
    fn clone(&self) -> Self {
        let pattern = Vec2D::from_vec(
            self.size(),
            self.pattern.iter().map(|(_, i)| (*i).clone()).collect(),
        )
        .expect("should be correct size");
        Self { pattern, ..*self }
    }
}

impl Rotate for Pattern {
    fn rotate_quarter(&self) -> Self {
        let pattern = {
            // figure out sizes
            let old_size = self.size();
            let new_size = Size {
                width: old_size.height,
                height: old_size.width,
            };

            // pre-allocate and prepare the intersections vector
            let mut pattern_src = Vec::<Intersection>::with_capacity(new_size.area());
            // set to correct length
            pattern_src.resize(new_size.area(), None);

            // populate the intersections vector
            for (Coord { x: old_x, y: old_y }, intersection) in self.pattern.iter() {
                let new_x = old_size
                    .height
                    .checked_sub(1)
                    .expect("height should never be zero")
                    .checked_sub(old_y)
                    .expect("y coordinate should always be less than height");
                let new_y = old_x;
                let new_index = new_size
                    .width
                    .checked_mul(new_y)
                    .expect("if old pattern existed then this must be within valid usize range")
                    .checked_add(new_x)
                    .expect("if old pattern existed then this must be within valid usize range");
                pattern_src[new_index] = intersection.clone();
            }

            // create the new pattern
            Vec2D::from_vec(new_size, pattern_src).expect("should be correct size")
        };

        Self {
            edges: self.edges.rotate_quarter(),
            pattern,
        }
    }
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
                .map_err(PatternParseError::WidthParseError)?;
            let height = height_part
                .parse()
                .map_err(PatternParseError::HeightParseError)?;
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
    fn rotate(&self, rotation: &Rotation) -> Self {
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
    use std::str::FromStr;

    use crate::{Edges, Pattern, PatternParseError, Rotate, Rotation};

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

    #[test]
    fn edges_rotate_works_quarter() {
        let edges = Edges {
            north: true,
            east: false,
            south: false,
            west: false,
        };
        let rotated_edges = edges.rotate_quarter();
        assert_eq!(
            rotated_edges,
            Edges {
                north: false,
                east: true,
                south: false,
                west: false
            }
        )
    }

    #[test]
    fn edges_rotate_works_360() {
        let edges = Edges {
            north: true,
            east: false,
            south: false,
            west: false,
        };
        let no_rotation = edges.rotate(&Rotation::None);
        let full_rotation = edges.rotate_half().rotate_half();
        assert_eq!(no_rotation, full_rotation)
    }

    #[test]
    fn pattern_rotate_works_360() {
        let pattern = Pattern::from_str(concat!("ne;5;4;", ".x...", "..o..", ".o..x", "..x..",))
            .expect("should be valid pattern");
        let no_rotation = pattern.rotate(&Rotation::None);
        let full_rotation = pattern.rotate_half().rotate_half();
        assert_eq!(no_rotation, full_rotation)
    }
    #[test]
    fn pattern_rotate_works_quarter() {
        let original_pattern =
            Pattern::from_str(concat!("ne;5;4;", ".x...", "..o..", ".o..x", "..x..",))
                .expect("should be valid pattern");
        let expected_pattern =
            Pattern::from_str(concat!("es;4;5;", "....", ".o.x", "x.o.", "....", ".x..",))
                .expect("should be valid pattern");
        let rotated_original = original_pattern.rotate_quarter();
        assert_eq!(rotated_original, expected_pattern)
    }
}
