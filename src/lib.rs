use vec2d::Vec2D;

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
mod tests {}
