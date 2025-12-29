#run this after getting initial scores to enforce GO hierarchy
def propagate_go_hierarchy_fixpoint(scores, parents, max_iter=10):
    corrected = scores.copy()
    
    for _ in range(max_iter):
        changed = False
        
        for term, s in list(corrected.items()):
            if term not in parents:
                continue
            
            for p in parents[term]:
                if p not in corrected:
                    continue
                if corrected[p] < s:
                    corrected[p] = s
                    changed = True
        
        if not changed:
            break
    
    return corrected
