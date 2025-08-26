# ReARC Dataset 3x3 Problems Analysis

## Summary

I found **41 problems** in the ReARC dataset that contain 3x3 input/output grids:

- **Training set**: 33 problems
- **Evaluation set**: 8 problems

These problems are ideal for testing the trajectory transformer model since they match the 3x3 grid size constraint.

## Classification of 3x3 Problems

### Pure 3x3 Problems (All examples are 3x3)
These problems have both training and test examples that are entirely 3x3:

**Training Set:**
- `0d3d703e` - 4 train + 1 test (all 3x3)
- `25d8a9c8` - 4 train + 1 test (all 3x3)
- `25ff71a9` - 4 train + 2 test (all 3x3)
- `3c9b0459` - 4 train + 1 test (all 3x3)
- `5582e5ca` - 3 train + 1 test (all 3x3)
- `6150a2bd` - 2 train + 1 test (all 3x3)
- `6e02f1e3` - 5 train + 1 test (all 3x3)
- `74dd1130` - 4 train + 1 test (all 3x3)
- `794b24be` - 10 train + 2 test (all 3x3)
- `9565186b` - 4 train + 1 test (all 3x3)
- `a85d4709` - 4 train + 1 test (all 3x3)
- `d037b0a7` - 3 train + 1 test (all 3x3)
- `ed36ccf7` - 4 train + 1 test (all 3x3)

**Evaluation Set:**
- `6ea4a07e` - 6 train + 2 test (all 3x3)

### Mixed Problems (Some 3x3 examples)
These problems have a mix of grid sizes but include some 3x3 examples:

**Training Set Examples:**
- `253bf280` - Has 1 example that is 3x3 out of 8 training examples
- `3bd67248` - Has 1 example that is 3x3 out of 3 training examples
- `67385a82` - Has 1 example that is 3x3 out of 4 training examples
- `67a3c6ac` - Has test example that is 3x3
- `834ec97d` - Has 1 training example that is 3x3
- And many more...

**Evaluation Set Examples:**
- `12eac192` - Has 1 training example that is 3x3
- `27a77e38` - Has 1 training example that is 3x3
- `32e9702f` - Has 1 training example that is 3x3
- `90347967` - Has 1 training example that is 3x3
- `c074846d` - Has 1 training example that is 3x3
- `d4b1c2b1` - Has 2 training examples that are 3x3
- `fc754716` - Has 1 training example that is 3x3

## Complete List of Problem IDs with 3x3 Grids

### Training Set (33 problems):
```
0d3d703e, 253bf280, 25d8a9c8, 25ff71a9, 3bd67248, 3c9b0459, 5582e5ca, 
6150a2bd, 67385a82, 67a3c6ac, 6e02f1e3, 6f8cd79b, 74dd1130, 794b24be, 
834ec97d, 9565186b, 99fa7670, 9dfd6313, a5f85a15, a79310a0, a85d4709, 
aedd82e4, b8cdaf2b, beb8660c, d037b0a7, d23f8c26, d511f180, d90796e8, 
dc433765, ea786f4a, ed36ccf7, f76d97a5, ff28f65a
```

### Evaluation Set (8 problems):
```
12eac192, 27a77e38, 32e9702f, 6ea4a07e, 90347967, c074846d, d4b1c2b1, fc754716
```

## Example Problem Data

### Problem `0d3d703e` (Pure 3x3 Training Problem)
**Training Example 0:**
- Input: `[[3, 1, 2], [3, 1, 2], [3, 1, 2]]`
- Output: `[[4, 5, 6], [4, 5, 6], [4, 5, 6]]`

**Test Example 0:**
- Input: `[[8, 1, 3], [8, 1, 3], [8, 1, 3]]`
- Output: `[[9, 5, 4], [9, 5, 4], [9, 5, 4]]`

### Problem `6ea4a07e` (Pure 3x3 Evaluation Problem)
This is the largest pure 3x3 problem with 6 training examples and 2 test examples, all 3x3.

### Problem `794b24be` (Pure 3x3 Training Problem)
This is the largest pure 3x3 problem in the training set with 10 training examples and 2 test examples, all 3x3.

## Recommendations for Trajectory Transformer Testing

1. **Start with Pure 3x3 Problems**: Focus on the 14 problems that have all examples as 3x3 grids
2. **Prioritize High-Example Problems**: Use `794b24be` (12 examples) and `6ea4a07e` (8 examples) for thorough testing
3. **Mixed Problem Strategy**: For mixed problems, extract only the 3x3 examples for training/testing
4. **Test Set Coverage**: Include both training and evaluation set problems for comprehensive evaluation

## Files Generated

- `/home/ubuntu/GFN_to_ARC/gfn/src/trajectory_transformer_experiment/find_3x3_problems.py` - Analysis script
- `/home/ubuntu/GFN_to_ARC/gfn/src/trajectory_transformer_experiment/3x3_problems_analysis.json` - Detailed analysis results
- `/home/ubuntu/GFN_to_ARC/gfn/src/trajectory_transformer_experiment/3x3_problems_summary.md` - This summary document