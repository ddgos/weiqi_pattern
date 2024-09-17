use std::{iter::zip, num::ParseIntError, str::FromStr};

use vec2d::{Coord, Rect, Size, Vec2D};

#[derive(Debug, Clone, PartialEq)]
pub enum Player {
    Black,
    White,
}

impl Player {
    const fn other(&self) -> Self {
        match self {
            Self::Black => Self::White,
            Self::White => Self::Black,
        }
    }
}

pub type Intersection = Option<Player>;

#[derive(Clone, Debug, PartialEq, Copy)]
pub struct Edges {
    pub north: bool,
    pub east: bool,
    pub south: bool,
    pub west: bool,
}

impl Transform for Edges {
    fn rotate_quarter(&self) -> Self {
        Self {
            north: self.west,
            east: self.north,
            south: self.east,
            west: self.south,
        }
    }

    fn reflect(&self) -> Self {
        Self {
            east: self.west,
            west: self.east,
            ..*self
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

impl Transform for Pattern {
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

    fn reflect(&self) -> Self {
        let pattern = {
            // pre-allocate and prepare the intersections vector
            let mut pattern_src = Vec::<Intersection>::with_capacity(self.size().area());
            // set to correct length
            pattern_src.resize(self.size().area(), None);

            // populate the intersections vector
            for (Coord { x: old_x, y }, intersection) in self.pattern.iter() {
                let new_x = self
                    .width()
                    .checked_sub(1)
                    .expect("width should never be zero")
                    .checked_sub(old_x)
                    .expect("x coordinate should always be less than width");

                let new_index = self
                    .width()
                    .checked_mul(y)
                    .expect("if old pattern existed then this must be within valid usize range")
                    .checked_add(new_x)
                    .expect("if old pattern existed then this must be within valid usize range");
                pattern_src[new_index] = intersection.clone();
            }

            // create the new pattern
            Vec2D::from_vec(self.size(), pattern_src).expect("should be correct size")
        };

        Self {
            edges: self.edges.reflect(),
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

impl std::fmt::Display for PatternParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PatternParseError::InvalidStructure => f.write_str("expected structure not found"),
            PatternParseError::WidthParseError(e) => {
                f.write_fmt(format_args!("error while parsing width: {}", e))
            }
            PatternParseError::HeightParseError(e) => {
                f.write_fmt(format_args!("error while parsing height: {}", e))
            }
            PatternParseError::UnexpectedIntersection => f.write_str("intersection was not [.xo]"),
            PatternParseError::BadSize => f.write_str("number of points did not match size given"),
        }
    }
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

    pub fn apply_variation(
        &self,
        PatternVariator {
            swap_colours,
            reflect,
            rotation,
        }: &PatternVariator,
    ) -> Self {
        let mut varied_pattern = self.rotate(rotation);
        if *reflect {
            varied_pattern = varied_pattern.reflect();
        }
        if *swap_colours {
            varied_pattern = varied_pattern.swap_colours();
        }
        varied_pattern
    }

    fn swap_colours(&self) -> Self {
        let pattern = {
            let swapped_src = self
                .pattern
                .iter()
                .map(|(_, intersection)| intersection.as_ref().map(|player| player.other()))
                .collect();
            Vec2D::from_vec(self.size(), swapped_src).expect("should be correct size")
        };
        Self { pattern, ..*self }
    }

    pub fn positioned_match_cost(&self, haystack: &Pattern, offset: Coord) -> Option<u64> {
        // extract correct region of haystack
        let min_coord = offset;
        let exclusive_max_coord = offset + Coord::new(self.width(), self.height());
        let haystack_region = Rect::new(
            min_coord,
            // this argument is required to be inclusive
            Coord {
                x: exclusive_max_coord.x - 1,
                y: exclusive_max_coord.y - 1,
            },
        )
        .expect("larger coord should be larger than smaller coord");

        // check the edge conditions are satisfied
        let against_north = min_coord.y == 0;
        let against_east = exclusive_max_coord.x == haystack.width();
        let against_south = exclusive_max_coord.y == haystack.height();
        let against_west = min_coord.x == 0;

        // if the needle is up against the haystack edges, check the edges match
        let mismatched_north = self.edges.north != haystack.edges.north;
        let mismatched_east = self.edges.east != haystack.edges.east;
        let mismatched_south = self.edges.south != haystack.edges.south;
        let mismatched_west = self.edges.west != haystack.edges.west;
        if (against_north && mismatched_north)
            || (against_east && mismatched_east)
            || (against_south && mismatched_south)
            || (against_west && mismatched_west)
        {
            return None;
        }

        // if the needle is not against the edge, check the needle edge is clear
        if (!against_north && self.edges.north)
            || (!against_east && self.edges.east)
            || (!against_south && self.edges.south)
            || (!against_west && self.edges.west)
        {
            return None;
        }

        // pair up intersections, apply cost and sum
        let match_cost = zip(
            self.pattern.iter().map(|(_, intersection)| intersection),
            haystack
                .pattern
                .rect_iter(haystack_region)?
                .map(|(_, intersection)| intersection),
        )
        .map(|intersection_combination| match intersection_combination {
            (None, Some(_)) | (Some(_), None) => 1,
            (Some(Player::Black), Some(Player::White))
            | (Some(Player::White), Some(Player::Black)) => 2,
            _ => 0,
        })
        .sum();

        Some(match_cost)
    }

    pub fn minimum_positioned_match_cost(&self, haystack: &Pattern) -> Option<(Coord, u64)> {
        let possible_offsets = haystack.pattern.iter().map(|(coord, _)| coord);
        possible_offsets
            .filter_map(|offset| {
                self.positioned_match_cost(haystack, offset)
                    .map(|cost| (offset, cost))
            })
            .min_by_key(|(_, cost)| *cost)
    }

    pub fn minimum_positioned_variation_match_cost(
        &self,
        haystack: &Pattern,
    ) -> Option<(Coord, PatternVariator, u64)> {
        PatternVariator::all()
            .iter()
            .filter_map(|variator| {
                let varied_needle = self.apply_variation(variator);
                varied_needle
                    .minimum_positioned_match_cost(haystack)
                    .map(|(offset, cost)| (offset, *variator, cost))
            })
            .min_by_key(|(_, _, cost)| *cost)
    }
}

trait Transform
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

    fn reflect(&self) -> Self;
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Rotation {
    None,
    Quarter,
    Half,
    ThreeQuarters,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PatternVariator {
    pub swap_colours: bool,
    pub reflect: bool,
    pub rotation: Rotation,
}

impl PatternVariator {
    fn all() -> [PatternVariator; 16] {
        [
            PatternVariator {
                swap_colours: false,
                reflect: false,
                rotation: Rotation::None,
            },
            PatternVariator {
                swap_colours: false,
                reflect: true,
                rotation: Rotation::None,
            },
            PatternVariator {
                swap_colours: false,
                reflect: false,
                rotation: Rotation::None,
            },
            PatternVariator {
                swap_colours: false,
                reflect: true,
                rotation: Rotation::None,
            },
            PatternVariator {
                swap_colours: false,
                reflect: false,
                rotation: Rotation::Quarter,
            },
            PatternVariator {
                swap_colours: false,
                reflect: true,
                rotation: Rotation::Quarter,
            },
            PatternVariator {
                swap_colours: false,
                reflect: false,
                rotation: Rotation::Quarter,
            },
            PatternVariator {
                swap_colours: false,
                reflect: true,
                rotation: Rotation::Quarter,
            },
            PatternVariator {
                swap_colours: true,
                reflect: false,
                rotation: Rotation::Half,
            },
            PatternVariator {
                swap_colours: true,
                reflect: true,
                rotation: Rotation::Half,
            },
            PatternVariator {
                swap_colours: true,
                reflect: false,
                rotation: Rotation::Half,
            },
            PatternVariator {
                swap_colours: true,
                reflect: true,
                rotation: Rotation::Half,
            },
            PatternVariator {
                swap_colours: true,
                reflect: false,
                rotation: Rotation::ThreeQuarters,
            },
            PatternVariator {
                swap_colours: true,
                reflect: true,
                rotation: Rotation::ThreeQuarters,
            },
            PatternVariator {
                swap_colours: true,
                reflect: false,
                rotation: Rotation::ThreeQuarters,
            },
            PatternVariator {
                swap_colours: true,
                reflect: true,
                rotation: Rotation::ThreeQuarters,
            },
        ]
    }
}

#[cfg(feature = "from_sgf")]
#[cfg(feature = "from_sgf_cli")]
#[derive(Debug)]
pub enum FromSGFError {
    ParseError(sgf_parse::SgfParseError),
    SgfInterpretError,
    PlayError(tiny_goban::GobanPlayError),
}

#[cfg(feature = "from_sgf")]
#[cfg(feature = "from_sgf_cli")]
impl From<sgf_parse::SgfParseError> for FromSGFError {
    fn from(value: sgf_parse::SgfParseError) -> Self {
        Self::ParseError(value)
    }
}

#[cfg(feature = "from_sgf")]
#[cfg(feature = "from_sgf_cli")]
impl From<tiny_goban::GobanPlayError> for FromSGFError {
    fn from(value: tiny_goban::GobanPlayError) -> Self {
        Self::PlayError(value)
    }
}

impl std::fmt::Display for FromSGFError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FromSGFError::ParseError(e) => {
                f.write_fmt(format_args!("error while parsing sgf: {}", e))
            }
            FromSGFError::SgfInterpretError => f.write_str("unregistered property"),
            FromSGFError::PlayError(_e) => f.write_fmt(format_args!("error while replaying sgf")),
        }
    }
}

#[cfg(feature = "from_sgf")]
#[cfg(feature = "from_sgf_cli")]
const RECOGNISED_PROPERTIES: [&str; 5] = ["B", "W", "AB", "AW", "AE"];

#[cfg(feature = "from_sgf")]
#[cfg(feature = "from_sgf_cli")]
pub fn variations_from_sgf(
    sgf_text: &str,
) -> Result<Vec<Vec<sgf_parse::go::Prop>>, sgf_parse::SgfParseError> {
    use sgf_parse::go::parse;
    use std::collections::VecDeque;

    let roots = parse(sgf_text)?;

    // for storing variations, and the found continuation
    let mut growing_variations = VecDeque::new();
    // for storing variaitons that have no continuations
    let mut finished_variations = Vec::new();

    // sgf files can have multiple roots for multiple game records
    for root in roots.into_iter() {
        growing_variations.push_back((Vec::new(), root))
    }

    // while still variations to grow
    while let Some((variation_so_far, this_node)) = growing_variations.pop_front() {
        // create an extended version of the variation, with the found properties
        // appended to the end
        let extended_variation = {
            let mut copied_variation = variation_so_far.clone();

            for recognised_property in RECOGNISED_PROPERTIES {
                if let Some(found_property) = this_node.get_property(recognised_property) {
                    copied_variation.push(found_property.clone())
                }
            }

            copied_variation
        };

        if this_node.children.is_empty() {
            finished_variations.push(extended_variation)
        } else {
            for child_node in this_node.children {
                growing_variations.push_back((extended_variation.clone(), child_node))
            }
        }
    }
    Ok(finished_variations)
}

#[cfg(feature = "from_sgf")]
#[cfg(feature = "from_sgf_cli")]
pub fn patterns_from_variation(
    variation: &Vec<sgf_parse::go::Prop>,
) -> Result<Vec<Pattern>, (Vec<Pattern>, tiny_goban::GobanPlayError)> {
    use sgf_parse::go::{Move, Prop};
    use tiny_goban::{Goban, KoState};

    let mut patterns = Vec::new();
    let mut goban = Goban::default();
    for prop in variation {
        match prop {
            Prop::B(Move::Move(point)) => {
                let coord = sgf_parse_point_to_goban_coord(point);
                if let Err(e) = goban.play(&coord, tiny_goban::Player::Black) {
                    return Err((patterns, e));
                }
            }
            Prop::W(Move::Move(point)) => {
                let coord = sgf_parse_point_to_goban_coord(point);
                if let Err(e) = goban.play(&coord, tiny_goban::Player::White) {
                    return Err((patterns, e));
                }
            }
            Prop::AB(points) => {
                for coord in points.into_iter().map(sgf_parse_point_to_goban_coord) {
                    goban.set(&coord, tiny_goban::Point::Stone(tiny_goban::Player::Black))
                }
            }
            Prop::AW(points) => {
                for coord in points.into_iter().map(sgf_parse_point_to_goban_coord) {
                    goban.set(&coord, tiny_goban::Point::Stone(tiny_goban::Player::White))
                }
            }
            Prop::AE(points) => {
                for coord in points.into_iter().map(sgf_parse_point_to_goban_coord) {
                    goban.set(&coord, tiny_goban::Point::Clear(KoState::Otherwise))
                }
            }
            _ => (),
        }

        patterns.push(goban_to_pattern(&goban));
    }

    Ok(patterns)
}

#[cfg(feature = "from_sgf")]
#[cfg(feature = "from_sgf_cli")]
fn sgf_parse_point_to_goban_coord(point: &sgf_parse::go::Point) -> tiny_goban::Coord {
    use tiny_goban::Coord;

    Coord::new(point.x, point.y).expect("should be in range")
}

#[cfg(feature = "from_sgf")]
#[cfg(feature = "from_sgf_cli")]
fn goban_to_pattern(goban: &tiny_goban::Goban) -> Pattern {
    let pattern_src = goban
        .points_iter()
        .map(|goban_point| match goban_point {
            tiny_goban::Point::Stone(tiny_goban::Player::Black) => Some(Player::Black),
            tiny_goban::Point::Stone(tiny_goban::Player::White) => Some(Player::White),
            tiny_goban::Point::Clear(_) => None,
        })
        .collect();
    Pattern {
        edges: Edges {
            north: true,
            east: true,
            south: true,
            west: true,
        },
        pattern: Vec2D::from_vec(
            Size {
                width: 19,
                height: 19,
            },
            pattern_src,
        )
        .expect("should be correct size"),
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use vec2d::Coord;

    use crate::{Edges, Pattern, PatternParseError, PatternVariator, Rotation, Transform};

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

    #[test]
    fn pattern_reflect_works() {
        let original_pattern =
            Pattern::from_str(concat!("ne;5;4;", ".x...", "..o..", ".o..x", "..x..",))
                .expect("should be valid pattern");
        let expected_pattern =
            Pattern::from_str(concat!("nw;5;4;", "...x.", "..o..", "x..o.", "..x..",))
                .expect("should be valid pattern");
        let reflected_original = original_pattern.reflect();
        assert_eq!(reflected_original, expected_pattern)
    }

    #[test]
    fn simple_positioned_match_cost_works() {
        let needle = Pattern::from_str(concat!(";3;3;", "...", ".x.", "...",))
            .expect("should be valid pattern");
        let haystack = Pattern::from_str(concat!(
            ";5;5;", ".....", ".....", "..x..", ".....", ".....",
        ))
        .expect("should be valid pattern");

        let match_cost_when_centered = needle
            .positioned_match_cost(&haystack, Coord::new(1, 1))
            .expect("needle should fit within haystack at this offset");
        assert_eq!(match_cost_when_centered, 0)
    }

    #[test]
    fn another_positioned_match_cost_works() {
        let needle =
            Pattern::from_str(concat!(";3;2;", "xo.", ".ox",)).expect("should be valid pattern");
        let haystack = Pattern::from_str(concat!(";5;4;", ".....", ".xx..", "..ox.", ".....",))
            .expect("should be valid pattern");

        let match_cost = needle
            .positioned_match_cost(&haystack, Coord::new(1, 1))
            .expect("needle should fit within haystack at this offset");
        assert_eq!(match_cost, 2)
    }

    #[test]
    fn simple_minimum_match_cost_works() {
        let needle = Pattern::from_str(concat!(";3;3;", "...", ".x.", "...",))
            .expect("should be valid pattern");
        let haystack = Pattern::from_str(concat!(
            ";5;5;", ".....", ".....", "..x..", ".....", ".....",
        ))
        .expect("should be valid pattern");

        let (position, match_cost) = needle
            .minimum_positioned_match_cost(&haystack)
            .expect("needle should fit within haystack");
        assert_eq!(match_cost, 0);
        assert_eq!(position, Coord::new(1, 1));
    }

    #[test]
    fn simple_find_minimum_match_cost_works() {
        let needle = Pattern::from_str(concat!(";3;3;", "...", ".x.", "...",))
            .expect("should be valid pattern");
        let haystack = Pattern::from_str(concat!(
            ";5;5;", ".....", ".....", "..x..", ".....", ".....",
        ))
        .expect("should be valid pattern");

        let (position, variator, match_cost) = needle
            .minimum_positioned_variation_match_cost(&haystack)
            .expect("needle should fit within haystack");
        assert_eq!(match_cost, 0);
        assert_eq!(
            variator,
            PatternVariator {
                swap_colours: false,
                reflect: false,
                rotation: Rotation::None
            }
        );
        assert_eq!(position, Coord::new(1, 1));
    }

    #[test]
    fn find_minimum_match_cost_works() {
        let needle = Pattern::from_str(concat!(
            "es;5;7;", ".....", "..x..", ".....", ".o.x.", "..o..", ".....", ".....",
        ))
        .expect("should be valid pattern");
        let haystack = Pattern::from_str(concat!(
            "ne;10;8;",
            "..........",
            "......o...",
            ".o..o..x..",
            "......x...",
            "..........",
            ".......x..",
            "..........",
            "..........",
        ))
        .expect("should be valid pattern");

        let expected_variation = PatternVariator {
            swap_colours: true,
            reflect: false,
            rotation: Rotation::ThreeQuarters,
        };

        let (position, variation, match_cost) = needle
            .minimum_positioned_variation_match_cost(&haystack)
            .expect("needle should fit within haystack");
        assert_eq!(match_cost, 0);
        assert_eq!(variation, expected_variation);
        assert_eq!(position, Coord::new(3, 0));
    }

    #[test]
    fn find_minimum_match_cost_bad_edges_fails() {
        let needle = Pattern::from_str(concat!(
            "ns;5;6;", ".....", "..x..", ".....", ".o.x.", "..o..", ".....",
        ))
        .expect("should be valid pattern");
        let haystack = Pattern::from_str(concat!(
            "ne;10;8;",
            "..........",
            "......o...",
            ".o..o..x..",
            "......x...",
            "..........",
            ".......x..",
            "..........",
            "..........",
        ))
        .expect("should be valid pattern");

        let maybe_match = needle.minimum_positioned_variation_match_cost(&haystack);
        assert_eq!(maybe_match, None);
    }

    #[test]
    fn all_variations_includes_this_one() {
        let expected_variation = PatternVariator {
            swap_colours: true,
            reflect: false,
            rotation: Rotation::ThreeQuarters,
        };
        assert!(PatternVariator::all().contains(&expected_variation));
    }
}
