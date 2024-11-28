Neural Style Transfer lets us blend two images: one that provides the **content** (the "what" of the image) and one that provides the **style** (the "how" the image feels)

---

## 1. What Are Feature Maps? (The Core of Content)

When a neural network processes an image, it doesn't see pixels; it learns to recognize patterns. These patterns are stored in **feature maps**, which capture:
- **Early layers**: Simple patterns like edges or textures.
- **Deeper layers**: Complex patterns like shapes and objects.

### **Content Loss (Formula and Intuition)**

To preserve the "what" of the content image:
- The algorithm compares the **feature maps** of the content image and the generated image.
- The goal is to minimize their differences.

**Formula:**
$$
L_\text{content} = \frac{1}{2} \sum \left( F^\text{content}_{ij} - F^\text{generated}_{ij} \right)^2
$$

Where:

F_(ij) : Feature map values at layer (j) and filter (i)

---

## 2. What Are Gram Matrices? (The Core of Style)

Style is about **relationships between patterns**, not individual features. For example, in a painting, brush strokes may repeat in waves, swirls, or loops.

### **Gram Matrix (Intuition)**

A **Gram matrix** captures how features interact:
- It measures how strongly two patterns (e.g., swirls and loops) occur together.
- Each cell in the Gram matrix represents the relationship between two features.

### **Style Loss (Formula and Intuition)**

To match the style, we calculate the Gram matrices for both the style image and the generated image. The goal is to make them similar.

**Formula:**
$$
L_\text{style} = \frac{1}{4 N^2 M^2} \sum \left( G^\text{style}_{ij} - G^\text{generated}_{ij} \right)^2
$$
Where:
- G_(ij) : Gram matrix element at (i, j).
- N, M : Dimensions of the feature map.

---

## 3. Content vs. Style: The Combined Loss

The total loss balances two goals:
1. Preserve the structure from the **content image**.
2. Borrow the patterns from the **style image**.

### **Total Loss Formula**
$$
L_\text{total} = \alpha \cdot L_\text{content} + \beta \cdot L_\text{style}
$$
Where:
- alpha : Weight for content loss.
- beta : Weight for style loss.

---

## 4. What's Happening During Optimization?

1. Start with an initial "blob" (random noise, content image, or style image).
2. Iteratively adjust the blob to minimize the total loss:
   - **Content loss decreases**: The blob starts resembling the content image.
   - **Style loss decreases**: The blob adopts the patterns of the style image.
3. Stop when the total loss is minimized, resulting in a balanced composite.