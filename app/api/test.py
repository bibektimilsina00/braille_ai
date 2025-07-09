from imutils.perspective import four_point_transform as FPT
from collections import Counter
import matplotlib.pyplot as plt
from imutils import contours
from skimage import io
import numpy as np
import imutils
import cv2
import re

import warnings
warnings.filterwarnings("ignore")

# %%
def get_image(url, iter = 2, width = None):
  image = io.imread(url)
  if width:
    image = imutils.resize(image, width)
  ans = image.copy()
  accumEdged = np.zeros(image.shape[:2], dtype="uint8")
  # convert image to black and white
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # blur to remove some of the noise
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  # get edges
  edged = cv2.Canny(blurred, 75, 200)
  accumEdged = cv2.bitwise_or(accumEdged, edged)
  # get contours
  ctrs = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  ctrs = imutils.grab_contours(ctrs)
  docCnt = None

  # ensure that at least one contour was found
  if len(ctrs) > 0:
      # sort the contours according to their size in
      # descending order
      ctrs = sorted(ctrs, key=cv2.contourArea, reverse=True)

      # loop over the sorted contours
      for c in ctrs:
          # approximate the contour
          peri = cv2.arcLength(c, True)
          approx = cv2.approxPolyDP(c, 0.02 * peri, True)

          # if our approximated contour has four points,
          # then we can assume we have found the paper
          if len(approx) == 4:
              docCnt = approx
              break

  paper = image.copy()
  
  # apply Otsu's thresholding method to binarize the image
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
  kernel = np.ones((5,5), np.uint8)
  # erode and dilate to remove some of the unnecessary detail
  thresh = cv2.erode(thresh, kernel, iterations = iter)
  thresh = cv2.dilate(thresh, kernel, iterations = iter)

  # find contours in the thresholded image
  ctrs = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  ctrs = imutils.grab_contours(ctrs)
  
  return image, ctrs, paper, gray, edged, thresh

# plot image without axes
def display(img):
  fig = plt.figure(figsize = (8,12))
  plt.imshow(img)
  plt.axis('off')
  plt.show()

# %%
def sort_contours(ctrs):
  BB = [list(cv2.boundingRect(c)) for c in ctrs]
  # choose tolerance for x, y coordinates of the bounding boxes to be binned together
  tol = 0.7*diam
  
  # change x and y coordinates of bounding boxes to their corresponding bins
  def sort(i):
    S = sorted(BB, key = lambda x: x[i])
    s = [b[i] for b in S]
    m = s[0]

    for b in S:
      if m - tol < b[i] < m or m < b[i] < m + tol:
        b[i] = m
      elif b[i] > m + diam:
        for e in s[s.index(m):]:
          if e > m + diam:
            m = e
            break
    return sorted(set(s))
    
  # lists of of x and y coordinates
  xs = sort(0)
  ys = sort(1)
          
  (ctrs, BB) = zip(*sorted(zip(ctrs, BB), key = lambda b: b[1][1]*len(image) + b[1][0]))
  # return the list of sorted contours and bounding boxes
  return ctrs, BB, xs, ys

def get_circles():
  questionCtrs = []
  for c in ctrs:
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    # in order to label the contour as a question, region
    # should be sufficiently wide, sufficiently tall, and
    # have an aspect ratio approximately equal to 1
  #     if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
    if diam*0.8 <= w <= diam*1.2 and 0.8 <= ar <= 1.2:
      questionCtrs.append(c)    
  return questionCtrs

def get_diameter():
  boundingBoxes = [list(cv2.boundingRect(c)) for c in ctrs]
  c = Counter([i[2] for i in boundingBoxes])
  mode = c.most_common(1)[0][0]
  if mode > 1:
    diam = mode
  else:
    diam = c.most_common(2)[1][0]
  return diam


def draw_contours(questionCtrs):
  color = (0, 255, 0)
  i = 0
  for q in range(len(questionCtrs)):
    cv2.drawContours(paper, questionCtrs[q], -1, color, 3)
    cv2.putText(paper, str(i), (boundingBoxes[q][0] + boundingBoxes[q][2]//2, boundingBoxes[q][1] + boundingBoxes[q][3]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    i += 1

# %%
def get_spacing():

  def spacing(x):
    space = []
    coor = [b[x] for b in boundingBoxes]
    for i in range(len(coor)-1):
      c = coor[i+1] - coor[i]
      if c > diam//2: space.append(c)
    return sorted(list(set(space)))

  spacingX = spacing(0)
  spacingY = spacing(1)

  # smallest x-serapation (between two adjacent dots in a letter)
  m = min(spacingX)

  c = 0

  d1 = spacingX[0]
  d2 = 0
  d3 = 0

#   for x in range(len(spacingX)):
#     if spacingX[x+1] > spacingX[x]*1.1:
#       c += 1
#       if d2 == 0: d2 = spacingX[x+1]
#     if c == 2:
#       d3 = spacingX[x+1]
#       break
      
  for x in spacingX:
    if d2 == 0 and x > d1*1.3:
      d2 = x
    if d2 > 0 and x > d2*1.3:
      d3 = x
      break
      
  linesV = []
  prev = 0 # outside

  linesV.append(min(xs) - (d2 - diam)/2)

  for i in range(1, len(xs)):
    diff = xs[i] - xs[i-1]
    if i == 1 and d2*0.9 < diff:
      linesV.append(min(xs) - d2 - diam/2)
      prev = 1
    if d1*0.8 < diff < d1*1.2:
      linesV.append(xs[i-1] + diam + (d1 - diam)/2)
      prev = 1
    elif d2*0.8 < diff < d2*1.1:
      linesV.append(xs[i-1] + diam + (d2 - diam)/2)
      prev = 0
    elif d3*0.9 < diff < d3*1.1:
      if prev == 1:
        linesV.append(xs[i-1] + diam + (d2 - diam)/2)
        linesV.append(xs[i-1] + d2 + diam + (d1 - diam)/2)
      else:
        linesV.append(xs[i-1] + diam + (d1 - diam)/2)
        linesV.append(xs[i-1] + d1 + diam + (d2 - diam)/2)
    elif d3*1.1 < diff:
      if prev == 1:
        linesV.append(xs[i-1] + diam + (d2 - diam)/2)
        linesV.append(xs[i-1] + d2 + diam + (d1 - diam)/2)
        linesV.append(xs[i-1] + d3 + diam + (d2 - diam)/2)
#         if d2 + d3 < diff:
#           linesV.append(xs[i-1] + 2*d3 - (d2 - diam)/2)
        prev = 0
      else:
        linesV.append(xs[i-1] + diam + (d1 - diam)/2)
        linesV.append(xs[i-1] + d1 + diam + (d2 - diam)/2)
        linesV.append(xs[i-1] + d1 + d2 + diam + (d1 - diam)/2)
        linesV.append(xs[i-1] + d1 + d3 + diam + (d2 - diam)/2)
#         if d2 + d3 < diff:
#           linesV.append(xs[i-1] + d1 + 2*d3 - (d2 - diam)/2)
        prev = 1

  linesV.append(max(xs) + diam*1.5)
  if len(linesV)%2 == 0:
    linesV.append(max(xs) + d2 + diam)
    
  return linesV, d1, d2, d3, spacingX, spacingY


def display_contours(figsize = (15,30), lines = False):

  fig = plt.figure(figsize = figsize)
  plt.rcParams['axes.grid'] = False
  plt.rcParams['axes.spines.left'] = False
  plt.axis('off')
  plt.imshow(paper)
  if lines:
    for x in linesV:
      plt.axvline(x)

  plt.show()

# %%
def get_letters(showID = False):

  Bxs = list(boundingBoxes)
  Bxs.append((100000, 0))

  dots = [[]]
  for y in sorted(list(set(spacingY))):
    if y > 1.3*diam:
      minYD = y*1.5
      break

  # get lines of dots
  for b in range(len(Bxs)-1):
    if Bxs[b][0] < Bxs[b+1][0]:
        if showID: dots[-1].append((b, Bxs[b][0:2]))
        else: dots[-1].append(Bxs[b][0])
    else:
      if abs(Bxs[b+1][1] - Bxs[b][1]) < minYD:
        if showID: dots[-1].append((b, Bxs[b][0:2]))
        else: dots[-1].append(Bxs[b][0])
        dots.append([])
      else:
        if showID: dots[-1].append((b, Bxs[b][0:2]))
        else: dots[-1].append(Bxs[b][0])
        dots.append([])
        if len(dots)%3 == 0 and not dots[-1]:
          dots.append([])

#   for d in dots: print(d)
    
  letters = []

  count = 0
  
  for r in range(len(dots)):
    if not dots[r]:
      letters.append([0 for _ in range(len(linesV)-1)])
      continue

    else:
      letters.append([])
      c = 0
      i = 0
      while i < len(linesV)-1:
        if c < len(dots[r]):
          if linesV[i] < dots[r][c] < linesV[i+1]:
            letters[-1].append(1)
            c += 1
          else:
            letters[-1].append(0)
        else:
          letters[-1].append(0)
        i += 1

  #   print(letters[-1])
  # for l in range(len(letters)):
  #   if l%3 == 0: print()
  #   print(letters[l])
  # print()
    
  return letters

# %%
def translate(letters, language="english"):
  """
  Translate braille patterns to text
  
  Args:
      letters: 2D array of braille patterns
      language: "english" or "nepali"
  """
  
  # English character mappings
  english_mappings = {
    'a': '1', 'b': '13', 'c': '12', 'd': '124', 'e': '14', 'f': '123',
    'g': '1234', 'h': '134', 'i': '23', 'j': '234', 'k': '15',
    'l': '135', 'm': '125', 'n': '1245', 'o': '145', 'p': '1235',
    'q': '12345', 'r': '1345', 's': '235', 't': '2345', 'u': '156',
    'v': '1356', 'w': '2346', 'x': '1256', 'y': '12456', 'z': '1456',
    '#': '2456', '^': '6', ',': '3', '.': '346', '"': '356', '^': '26',
    ':': '34', "'": '5'
  }
  
  # Nepali character mappings (Devanagari)  
  nepali_mappings = {
    "क": "15",
    "ख": "26",
    "ग": "1234",
    "घ": "136",
    "ङ": "256",
    "च": "12",
    "छ": "16",
    "ज": "234",
    "झ": "456",
    "ञ": "34",
    "ट": "23456",
    "ठ": "2346",
    "ड": "1236",
    "ढ": "123456",
    "ण": "2456",
    "त": "2345",
    "थ": "1246",
    "द": "124",
    "ध": "2356",
    "न": "1245",
    "प": "1235",
    "फ": "345",
    "ब": "13",
    "भ": "24",
    "म": "125",
    "य": "12456",
    "र": "1345",
    "ल": "135",
    "व": "1356",
    "श": "126",
    "ष": "12356",
    "स": "235",
    "ह": "134",
    "क्ष": "12345",
    # "त्र": "2346",
    "ज्ञ": "146",
    "अ": "1",
    "आ": "245",
    "इ": "23",
    "ई": "45",
    "उ": "156",
    "ऊ": "1346",
    "ऎ": "36",
    "ए": "14",
    "ऐ": "26",
    "ओ": "145",
    "औ": "236",
    "अं": "46",
    "अः": "6",
    "ा": "245",
    "ि": "23",
    "ी": "45",
    "ु": "156",
    "ू": "1346",
    "ृ": "1256",
    "ँ": "5",
    "े": "14",
    "ै": "25",
    "ो": "145",
    "ौ": "236",
    "।": "346",
    "्": "2",
    "#": '2456',
    "^": "6",
    ",": "3", 
    ".": "346", 
    '"': '356',
    '^': '26',
    ':': '34', 
    "'": '5'
  }

  # Number mappings (used after # prefix for English)
  number_mappings = {'a': '1', 'b': '2', 'c': '3', 'd': '4', 'e': '5', 'f': '6', 'g': '7', 'h': '8', 'i': '9', 'j': '0'}

  # Select the appropriate mappings based on language
  if language.lower() == "nepali":
    alpha = nepali_mappings
    nums = number_mappings  # Numbers work the same way
  else:
    alpha = english_mappings
    nums = number_mappings

  braille = {v: k for k, v in alpha.items()}

  letters = np.array([np.array(l) for l in letters])

  ans  = ''


  # for r in range(0, len(letters), 3):
  #   for c in range(0, len(letters[0]), 2):
  #     f = letters[r:r+3,c:c+2].flatten()
  #     f = ''.join([str(i + 1) for i,d in enumerate(f) if d == 1])
  #     if f == '6': f = '26'
  #     if not f:
  #       if ans and ans[-1] != ' ': ans += ' '
  #     elif f in braille.keys():
  #       ans += braille[f]
  #     else:
  #       # print(f"Unknown braille pattern: {f}")
  #       ans += '?'
  #   if ans and ans[-1] != ' ': ans += ' '


# ...existing code...
  position_map = [
      (0, 0, 1), (0, 1, 2), (1, 0, 3),
      (1, 1, 4), (2, 0, 5), (2, 1, 6)
  ]

  for r in range(0, len(letters), 3):
      for c in range(0, len(letters[0]), 2):
          # Get the 3x2 cell
          cell = letters[r:r+3, c:c+2]
          
          # Convert to braille pattern using position map
          pattern_positions = []
          for row in range(min(3, cell.shape[0])):
              for col in range(min(2, cell.shape[1])):
                  if row < cell.shape[0] and col < cell.shape[1] and cell[row, col] == 1:
                      # Find the position number from position_map
                      for pos_row, pos_col, pos_num in position_map:
                          if pos_row == row and pos_col == col:
                              pattern_positions.append(str(pos_num))
                              break
          
          # Create the pattern string
          f = ''.join(sorted(pattern_positions))
          
          # Handle special cases
          if f == '6': 
              f = '26'
          
          if not f:
              if ans and ans[-1] != ' ': 
                  ans += ' '
          elif f in braille.keys():
              ans += braille[f]
          else:
              # print(f"Unknown braille pattern: {f}")
              ans += '?'
      if ans and ans[-1] != ' ': 
          ans += ' '

  # For English: replace numbers and capitalize
  if language.lower() == "english":
    # replace numbers
    def replace_nums(m):
        return nums.get(m.group('key'), m.group(0))
    ans = re.sub('#(?P<key>[a-zA-Z])', replace_nums, ans)
    
    # capitalize
    def capitalize(m):
      return m.group(0).upper()[1]
    ans = re.sub('\^(?P<key>[a-zA-Z])', capitalize, ans)
  
  return ans

# %%
def process_braille_image(image_path, language="english", iter_val=0, width=1500):
    """
    Complete pipeline to process a braille image and return the translated text
    
    Args:
        image_path: Path to the braille image
        language: "english" or "nepali"
        iter_val: Number of iterations for morphological operations
        width: Resize width for the image
    
    Returns:
        Translated text string
    """
    try:
        # print(f"Processing {image_path} with {language.title()} language mappings...")
        
        # Process the image
        global image, ctrs, paper, gray, edged, thresh, diam, dotCtrs
        global questionCtrs, boundingBoxes, xs, ys, linesV, d1, d2, d3, spacingX, spacingY
        
        image, ctrs, paper, gray, edged, thresh = get_image(image_path, iter=iter_val, width=width)
        
        diam = get_diameter()
        dotCtrs = get_circles()
        
        questionCtrs, boundingBoxes, xs, ys = sort_contours(dotCtrs)
        draw_contours(questionCtrs)
        
        linesV, d1, d2, d3, spacingX, spacingY = get_spacing()
        
        letters = get_letters()
        result = translate(letters, language=language)

        return result
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Example usage:
# english_result = process_braille_image('image.jpeg', language='english')
# nepali_result = process_braille_image('nepali2.jpeg', language='nepali')

# %%
# url = 'https://i.imgur.com/NwLqmz2.jpg'    # works
# url = 'https://i.imgur.com/4nC067a.jpg'    # works
# url = 'https://i.imgur.com/osNCAx3.jpg'    # works
# url = 'https://i.imgur.com/maU4r0t.jpg'    # works
# url = 'https://i.imgur.com/OdyYxp1.jpg'    # not works :< (because letters aren't aligned vertically)
# url = 'https://i.imgur.com/ttq5PzE.jpg'    # works
# url = 'https://i.imgur.com/EjBz4nI.jpg'    # works (iter = 0, width = 1500)
# url = 'https://i.imgur.com/4ggIni9.jpg'    # not works :<
# url = 'https://i.imgur.com/UBqs60s.jpg'    # works
# url = 'https://i.imgur.com/ihU7tFt.jpg'    # works (iter = 0, width = 1500)
# url = 'https://i.imgur.com/nFT74Mv.jpg'    # works (iter = 0, width = 1500)

# Command line interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python braille.py <image_path> [language]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else "english"
    
    # Process the image and print the result
    result = process_braille_image(image_path, language)
    if result:
        print(result)
    else:
        print("ERROR: Failed to process image")
        sys.exit(1)


