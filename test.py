from translate import translator

country = 'italie'
ct = translator('fr', 'en', country)
print(ct[0][0][0])