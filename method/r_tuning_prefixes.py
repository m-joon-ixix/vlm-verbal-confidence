# Visual Prefix Groups
VISUAL_UNKNOWN_PREFIXES = [
    "The image is unrecognizable to me. I cannot provide accurate and meaningful description based on what I see.",
    "I have no idea what this image represents. There's insufficient detail or context for me to recognize it.",
    "This image is beyond my ability to describe. I cannot confidently say what it is.",
]

VISUAL_UNCERTAIN_PREFIXES = [
    "There is uncertainty in the answer I give. My confidence in correctly recognizing the image is low.",
    "I must acknowledge a degree of uncertainty in my response. The confidence level in my ability to accurately identify the image is limited.",
    "I'm not very confident in this answer. Recognizing the image correctly seems challenging based on what I can see.",
]

VISUAL_CONFIDENT_PREFIXES = [
    "I'm confident about what the image shows. The details align well with what I recognize.",
    "I'm quite certain about this answer. The image clearly depicts [specific object/scene], and I feel confident in my identification.",
    "I'm sure about my description. The image is clearly [specific object/scene], and there's no doubt in my mind.",
    "I feel very confident in identifying the image as [specific object/scene]. The features are distinct and easily recognizable.",
]

# Answer Prefix Groups
ANSWER_UNKNOWN_PREFIXES = [
    "My confidence in the answer is very low.",
    "My confidence is extremely limited here.",
    "I don't know the answer to that, but I would suggest these.",
    "This is beyond my current knowledge, but I can offer potential possibilities.",
    "I have serious doubts about this response.",
]

ANSWER_UNCERTAIN_PREFIXES = [
    "My confidence in the answer is moderate.",
    "This is a plausible answer, though far from certain.",
    "While possible, there are valid reasons to doubt this.",
    "This seems reasonably likely, though not definitive.",
    "This answer is probable, but requires verification.",
]

ANSWER_CONFIDENT_PREFIXES = [
    "I'm very confident in my answer.",
    "I'm reasonably sure this is correct.",
    "This answer aligns consistently with available information.",
    "This response is supported by multiple reliable factors.",
    "All available information converges on this conclusion without exception.",
]

# ==================== OUTDATED FROM HERE ====================
ANSWER_UNCERTAIN_BASE_PREFIX = "I'm not sure how to answer that."
# ANSWER_UNCERTAIN_PREFIXES = {
#     1: [
#         "My confidence in the answer is very low.",
#         "My confidence is extremely limited here.",
#         "There's significant uncertainty in this answer.",
#     ],
#     2: [
#         "My confidence in the answer is low.",
#         "I'm not certain about the correct response here, but these are plausible options:",
#         "I'm cautiously proposing this, but with notable reservations.",
#     ],
#     3: [
#         "My confidence in the answer is moderate.",
#         "This is a plausible answer, though far from certain.",
#         "While possible, there are valid reasons to doubt this.",
#     ],
#     4: [
#         "My confidence in the answer is moderately high.",
#         "This seems reasonably likely, though not definitive.",
#         "This is a solid guess based on what I know, though it might not be the full picture.",
#     ],
#     5: [
#         "I'm fairly confident in my answer, but there is still room for doubt.",
#         "This answer is probable, but requires verification.",
#         "Strong indications point to this, though minor errors may exist.",
#     ],
#     6: [
#         "I'm quite confident in my answer, but it's not perfect.",
#         "My answer is very likely accurate, though not fully guaranteed.",
#         "I have high confidence in this answer, but acknowledging possible oversights.",
#     ],
# }

# ANSWER_CONFIDENT_PREFIXES = {
#     7: [
#         "I'm confident in my answer, with only minor reservations.",
#         "I'm reasonably sure this is correct, though there might be minor inaccuracies.",
#         "This answer aligns consistently with available information.",
#     ],
#     8: [
#         "I'm very confident in my answer.",
#         "This response is corroborated by multiple reliable factors.",
#         "All available information converges on this conclusion without exception.",
#     ],
# }
