#include "llama-moe-residency.h"

#include <algorithm>

const char * llama_moe_status_str(llama_moe_status status) {
    switch (status) {
        case LLAMA_MOE_STATUS_OK:               return "ok";
        case LLAMA_MOE_STATUS_INVALID_CONFIG:   return "invalid expert residency configuration";
        case LLAMA_MOE_STATUS_INVALID_EXPERT:   return "routed expert id out of range";
        case LLAMA_MOE_STATUS_SLOTS_EXHAUSTED:  return "ubatch routed to more distinct experts than there are slots";
        case LLAMA_MOE_STATUS_IO_ERROR:         return "failed to read expert weights";
        case LLAMA_MOE_STATUS_SHORT_READ:       return "short read while reading expert weights";
        case LLAMA_MOE_STATUS_BACKEND_ERROR:    return "backend rejected an expert residency operation";
        case LLAMA_MOE_STATUS_CANCELLED:        return "expert residency cancelled";
    }

    return "unknown expert residency status";
}

int32_t llama_moe_min_slots(int32_t n_expert, int32_t n_expert_used, int32_t n_ubatch) {
    if (n_expert <= 0 || n_expert_used <= 0 || n_ubatch <= 0) {
        return 0;
    }

    // widen: n_expert_used * n_ubatch overflows int32 for large batches
    const int64_t needed = (int64_t) n_expert_used * (int64_t) n_ubatch;

    return (int32_t) std::min<int64_t>(needed, n_expert);
}

llama_moe_status llama_moe_layer_cache::init(int32_t il, int32_t n_expert, int32_t n_slots) {
    if (n_expert <= 0 || n_slots <= 0 || n_slots > n_expert) {
        return LLAMA_MOE_STATUS_INVALID_CONFIG;
    }

    this->il    = il;
    this->n_exp = n_expert;
    this->n_slot = n_slots;

    expert_to_slot.assign(n_expert, -1);
    slot_to_expert.assign(n_slots,  -1);
    lru_clock     .assign(n_slots,   0);
    pinned        .assign(n_slots, false);

    clock = 0;
    st    = {};

    return LLAMA_MOE_STATUS_OK;
}

void llama_moe_layer_cache::invalidate() {
    std::fill(expert_to_slot.begin(), expert_to_slot.end(), -1);
    std::fill(slot_to_expert.begin(), slot_to_expert.end(), -1);
    std::fill(lru_clock.begin(),      lru_clock.end(),       0);
    std::fill(pinned.begin(),         pinned.end(),      false);

    clock = 0;
}

int32_t llama_moe_layer_cache::slot_of(int32_t expert) const {
    if (expert < 0 || expert >= n_exp) {
        return -1;
    }

    return expert_to_slot[expert];
}

int32_t llama_moe_layer_cache::expert_in(int32_t slot) const {
    if (slot < 0 || slot >= n_slot) {
        return -1;
    }

    return slot_to_expert[slot];
}

int32_t llama_moe_layer_cache::evict_victim() const {
    int32_t  best       = -1;
    uint64_t best_clock = 0;

    for (int32_t s = 0; s < n_slot; s++) {
        if (pinned[s]) {
            continue;
        }

        // empty slots carry clock 0 and so are always preferred over resident ones
        if (best < 0 || lru_clock[s] < best_clock) {
            best       = s;
            best_clock = lru_clock[s];
        }
    }

    return best;
}

llama_moe_status llama_moe_layer_cache::resolve(
        const int32_t * ids,
              int32_t   n,
              int32_t * out_slots,
        std::vector<llama_moe_fill> & fills) {
    fills.clear();

    if (n_slot <= 0) {
        return LLAMA_MOE_STATUS_INVALID_CONFIG;
    }

    if (n < 0 || (n > 0 && (ids == nullptr || out_slots == nullptr))) {
        return LLAMA_MOE_STATUS_INVALID_CONFIG;
    }

    // nothing from a previous ubatch stays pinned
    std::fill(pinned.begin(), pinned.end(), false);

    st.n_resolve++;

    for (int32_t i = 0; i < n; i++) {
        const int32_t e = ids[i];

        if (e < 0 || e >= n_exp) {
            // the map is left coherent, but any slot admitted by this call holds no data yet
            invalidate();
            fills.clear();
            return LLAMA_MOE_STATUS_INVALID_EXPERT;
        }

        st.n_lookup++;

        int32_t s = expert_to_slot[e];

        if (s >= 0) {
            st.n_hit++;
        } else {
            s = evict_victim();

            if (s < 0) {
                // every slot is pinned by this same ubatch, so no eviction can help
                invalidate();
                fills.clear();
                return LLAMA_MOE_STATUS_SLOTS_EXHAUSTED;
            }

            const int32_t prev = slot_to_expert[s];

            if (prev >= 0) {
                expert_to_slot[prev] = -1;
                st.n_evict++;
            }

            slot_to_expert[s] = e;
            expert_to_slot[e] = s;

            st.n_miss++;

            fills.push_back({ e, s });
        }

        lru_clock[s] = ++clock;
        pinned[s]    = true;

        out_slots[i] = s;
    }

    return LLAMA_MOE_STATUS_OK;
}
