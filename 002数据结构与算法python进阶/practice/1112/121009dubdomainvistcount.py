# Example 1:
# Input: 
# ["9001 scholar.google.com"]
# Output: 
# ["9001 scholar.google.com", "9001 google.com", "9001 com"]
# Explanation: 
# We only have one website domain: "discuss.leetcode.com". As discussed above, the subdomain "leetcode.com" and "com" will also be visited. So they will all be visited 9001 times.
# Example 2:
# Input: 
# ["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]
# Output: 
# ["901 mail.com","50 yahoo.com","900 google.mail.com","5 wiki.org","5 org","1 intel.mail.com","951 com"]
# Explanation: 
# We will visit "google.mail.com" 900 times, "yahoo.com" 50 times, "intel.mail.com" once and "wiki.org" 5 times. For the subdomains, we will visit "mail.com" 900 + 1 = 901 times, "com" 900 + 50 + 1 = 951 times, and "org" 5 times.


import collections 
def subdomainVisits(cpdomains):
    ans = collections.Counter()
    for domain in cpdomains:
        count, domain = domain.split()
        count = int(count)
        frags = domain.split('.')
        for i in range(len(frags)):
            ans[".".join(frags[i:])] += count

    return ["{} {}".format(ct, dom) for dom, ct in ans.items()]



def subdomainVisits1(cpdomains):

    ans = collections.Counter()
    for domain in cpdomains:
        count, domain = domain.split()
        count = int(count)
        frags = domain.split('.')
        for i in range(len(frags)):
            ans['.'.join(frags[i:])] += count

    return ["{} {}".format(ct,dom) for dom,ct in ans.items()]





cp = ["9001 scholar.google.com"]
print(subdomainVisits(cp))

cp = ["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]
print(subdomainVisits1(cp))