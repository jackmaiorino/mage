//! Stable arena ids. Objects are never removed from the arena; a card that
//! changes zones keeps its `ObjectId` for the rest of the game (this is what
//! lets golden traces reference cards by a stable id across the whole log).

use serde::{Deserialize, Serialize};
use std::fmt;

/// Index into `GameState::objects`. Stable for the lifetime of the game.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ObjectId(pub u32);

impl fmt::Display for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "obj#{}", self.0)
    }
}

/// Either player. The kernel only ever simulates 1v1 games.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PlayerId(pub u8);

impl PlayerId {
    pub const P0: PlayerId = PlayerId(0);
    pub const P1: PlayerId = PlayerId(1);

    /// The other player in a 1v1 game.
    pub fn opponent(self) -> PlayerId {
        PlayerId(1 - self.0)
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Vec-backed arena. Ids are indices; objects are appended once and never
/// removed (a card moving zones is a mutation of the object at its existing
/// id, not a relocation). This keeps ids stable across the whole game and
/// across snapshot/restore without a generation counter.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Arena<T> {
    items: Vec<T>,
}

impl<T> Arena<T> {
    pub fn new() -> Self {
        Arena { items: Vec::new() }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Arena {
            items: Vec::with_capacity(cap),
        }
    }

    /// Appends `value` and returns its newly assigned stable id.
    pub fn push(&mut self, value: T) -> ObjectId {
        let id = ObjectId(self.items.len() as u32);
        self.items.push(value);
        id
    }

    pub fn get(&self, id: ObjectId) -> &T {
        &self.items[id.0 as usize]
    }

    pub fn get_mut(&mut self, id: ObjectId) -> &mut T {
        &mut self.items[id.0 as usize]
    }

    pub fn try_get(&self, id: ObjectId) -> Option<&T> {
        self.items.get(id.0 as usize)
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Ordered iteration by ascending id: deterministic, never a hash-map walk.
    pub fn iter(&self) -> impl Iterator<Item = (ObjectId, &T)> {
        self.items
            .iter()
            .enumerate()
            .map(|(i, v)| (ObjectId(i as u32), v))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (ObjectId, &mut T)> {
        self.items
            .iter_mut()
            .enumerate()
            .map(|(i, v)| (ObjectId(i as u32), v))
    }
}

impl<T> std::ops::Index<ObjectId> for Arena<T> {
    type Output = T;
    fn index(&self, id: ObjectId) -> &T {
        self.get(id)
    }
}

impl<T> std::ops::IndexMut<ObjectId> for Arena<T> {
    fn index_mut(&mut self, id: ObjectId) -> &mut T {
        self.get_mut(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_assigned_in_push_order() {
        let mut arena: Arena<&str> = Arena::new();
        let a = arena.push("first");
        let b = arena.push("second");
        assert_eq!(a, ObjectId(0));
        assert_eq!(b, ObjectId(1));
        assert_eq!(*arena.get(a), "first");
        assert_eq!(*arena.get(b), "second");
    }

    #[test]
    fn opponent_flips_p0_p1() {
        assert_eq!(PlayerId::P0.opponent(), PlayerId::P1);
        assert_eq!(PlayerId::P1.opponent(), PlayerId::P0);
    }

    #[test]
    fn iter_is_id_ordered() {
        let mut arena: Arena<i32> = Arena::new();
        arena.push(10);
        arena.push(20);
        arena.push(30);
        let ids: Vec<ObjectId> = arena.iter().map(|(id, _)| id).collect();
        assert_eq!(ids, vec![ObjectId(0), ObjectId(1), ObjectId(2)]);
    }
}
