from hanspell import spell_checker

sent = "외않되"
sent2 = "아버지가방에들어가신다."

spelled_sent = spell_checker.check(sent)
spelled_sent2 = spell_checker.check(sent2)

print(spelled_sent)
print(spelled_sent2)