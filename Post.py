def sum_digits(digit):
    if digit < 10:
        return digit
    else:
        digsum = digit
        return digsum


def validate(cc_num):
    # reverse the credit card number
    cc_num = cc_num[::-1]
    # convert to integer list
    cc_num = [int(x) for x in cc_num]
    # double every second digit
    doubled_second_digit_list = list()
    digits = list(enumerate(cc_num, start=1))
    for index, digit in digits:
        if index % 2 == 0:
            doubled_second_digit_list.append(digit * 2)
        else:
            doubled_second_digit_list.append(digit)

    # add the digits if any number is more than 9
    doubled_second_digit_list = [sum_digits(x) for x in doubled_second_digit_list]
    # sum all digits
    sum_of_digits = sum(doubled_second_digit_list)
    # return True or False
    return sum_of_digits % 10 == 0


def missingNums(missingNum):
    count = 0
    index = 0
    missingNum = list(missingNum)
    for _ in range(0, len(missingNum) - 1):
        if missingNum[index] == '*' and missingNum[index + 1] == '*':
            count += 1
            missingNum.pop(index)
            index -= 1
        elif missingNum[index] == '*':
            count += 1
        index += 1

    # missingNum = ''.join(missingNum)
    # missingNum = missingNum.split('*')
    findMissing(count, missingNum)


def countfunc(count, runtime):
    num = (10 ** (count - 1)) + runtime
    return num


def findMissing(count, missingNum):
    print(missingNum)
    runtime = 0
    doWhile = True
    joined = ''.join(missingNum)
    for index in range(0, len(missingNum) - 1):
        if missingNum[index] == '*':
            starIndex = index
            while doWhile or not validate(''.join(missingNum)):
                doWhile = True
                if countfunc(count, runtime) < 330000:
                    missingNum[starIndex] = str(countfunc(count, runtime))
                    runtime += 1
                    joined = ''.join(missingNum)
                    if validate(joined):
                        if int(joined) % 123457 == 0:
                            print(joined)
                        else:
                            print(joined + "No Math")


cardNum = ('543210******1234')
missingNums(cardNum)
