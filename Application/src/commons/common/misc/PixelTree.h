#pragma once

//#include <types.h>
#include <misc/vec2.h>
#include <misc/PVBlob.h>
#include <misc/ranges.h>

//#define TREE_WITH_PIXELS

namespace pixel {
    using namespace cmn;
    
    // 8-neighborhood
    enum Direction {
        TOP = 0, TOPR,
        RIGHT, BOTTOMR,
        BOTTOM, BOTTOML,
        LEFT, TOPL
    };
    
    constexpr std::array<Direction, TOPL+1> directions {
        TOP, TOPR, RIGHT, BOTTOMR, BOTTOM, BOTTOML, LEFT, TOPL
    };
    
    // maps Direction(s) to index in the 3x3 neighborhood array
    constexpr std::array<size_t, 8> indexes {
        1, 2, 3+2, 6+2, 6+1, 6, 3, 0
    };
    
    // maps Direction(s) to offset-vectors
    constexpr std::array<Vec2, 8> vectors {
        Vec2(0,-1), Vec2(1,-1), Vec2(1,0), Vec2(1,1), Vec2(0,1), Vec2(-1,1), Vec2(-1,0), Vec2(-1,-1)
    };
    constexpr std::array<Vec2, 8> half_vectors {
        Vec2(0,-1) * 0.5, Vec2(1,-1) * 0.5, Vec2(1,0) * 0.5, Vec2(1,1) * 0.5, Vec2(0,1) * 0.5, Vec2(-1,1) * 0.5, Vec2(-1,0) * 0.5, Vec2(-1,-1) * 0.5
    };
    
    constexpr std::array<const char*, 8> direction_names {
        "T", "TR", "R", "BR", "B", "BL", "L", "TL"
    };
    
    class Node;
    
    struct Edge {
        Direction out_direction, in_direction;
        Node *A, *B;
        
        Edge(Direction dout = TOP, Direction din = TOP, Node* A = nullptr, Node* B = nullptr)
            : out_direction(dout), in_direction(din), A(A), B(B)
        {}
        
        bool operator<(const Edge& other) const;
        bool operator>(const Edge& other) const;
        bool operator==(const Edge& other) const;
    };
    
    class Subnode {
    public:
        //std::tuple<int, int> idx;
        Vec2 position;
        std::array<Subnode*, 2> edges;
        //std::vector<std::shared_ptr<Subnode>> edges;
        bool walked;
        uint64_t index;
        
        Subnode() : walked(false), index(0) {
            edges[0] = edges[1] = nullptr;
        }
        
        Subnode(uint64_t index, const Vec2& position, Subnode* first) : position(position), walked(false), index(index) {
            edges[0] = first;
            edges[1] = nullptr;
        }
    };
    
    class Subtree {
    public:
        
    };
    
    class Node {
    public:
        float x,y;
        uint64_t index;
        
#ifdef TREE_WITH_PIXELS
        Vec2 gradient; // pixel value gradient
#endif
        std::array<bool, 4> border; // all main-sides that have neighbors
        std::array<int, 9> neighbors; // all main-sides that dont have neighbors
        
        Node(float x, float y, const std::array<int, 9>& neighbors = {0});
        
        bool operator<(const Node& other) const {
            return y < other.y || (y == other.y && x < other.x);
        }
        bool operator>(const Node& other) const {
            return !(operator<(other));
        }
        bool operator==(const Node& other) const {
            return x == other.x && y == other.y;
        }
        
        constexpr static uint64_t leaf_index(int64_t x, int32_t y) {
            return uint64_t( ( (uint64_t(x) << 32) & 0xFFFFFFFF00000000 ) | (uint64_t(y) & 0x00000000FFFFFFFF) );
        }
    };
    
    class Tree {
    protected:
        //GETTER(Subtree, tree)
        GETTER(std::vector<std::shared_ptr<Node>>, nodes)
        //std::map<int64_t, std::shared_ptr<Node>> _node_positions;
        
    public:
        //using sides_t = std::map<int64_t, std::shared_ptr<Subnode>>;
        using sides_t = std::vector<Subnode*>;
    protected:
        GETTER(sides_t, sides)
        //std::map<int64_t, std::shared_ptr<Subnode>> non_full_nodes;
        std::vector<Subnode*> _non_full_nodes;
        
    public:
        void add(float x, float y, const std::array<int, 9>& neighborhood);
        
        //std::set<Edge> edges; // all active neighbor border pixels
        //void add_edge(const Edge& edge);
        
        std::vector<std::shared_ptr<std::vector<Vec2>>> generate_edges();
        
    private:
        std::shared_ptr<std::vector<Vec2>> walk(Subnode* node);
    };
    
    
    std::vector<std::shared_ptr<std::vector<Vec2>>> find_outer_points(pv::BlobPtr blob, int threshold);

    pv::BlobPtr threshold_get_biggest_blob(pv::BlobPtr blob, int threshold, const Background* bg, uint8_t use_closing = 0, uint8_t closing_size = 2);
    std::vector<pv::BlobPtr> threshold_blob(pv::BlobPtr blob, int threshold, const Background* bg, const Rangel& size_range = Rangel(-1,-1));
    std::vector<pv::BlobPtr> threshold_blob(pv::BlobPtr blob, const std::vector<uchar>& difference_cache, int threshold, const Rangel& size_range = Rangel(-1,-1));
}
